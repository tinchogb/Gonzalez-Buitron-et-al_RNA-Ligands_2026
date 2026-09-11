import ast
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

AGG_OPS = (
    "proportion", "count", "nunique", "sum", "mean",
    "median", "min", "max", "prop", "prop_sum", "prop_count", "prop_nunique"
)


def _count_items(val: Any) -> int:
    """Parse list, string, or set representation to obtain total item count."""
    if pd.isna(val):
        return 0
    if isinstance(val, (list, set, tuple)):
        return len(val)
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "[]", "{}", "set()", "None", "nan"):
            return 0
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, set, tuple)):
                    return len(parsed)
            except Exception:
                pass
        return len([item.strip() for item in val.split(",") if item.strip()])
    return 0


def _parse_bin_condition(bin_str: str) -> Callable[[Any], bool]:
    """Parse a bin string (e.g. '0', '>5', '<=2', '1-3') into a boolean evaluator function."""
    s = str(bin_str).strip()
    if s.startswith(">="):
        val = float(s[2:].strip())
        return lambda x: x >= val
    elif s.startswith(">"):
        val = float(s[1:].strip())
        return lambda x: x > val
    elif s.startswith("<="):
        val = float(s[2:].strip())
        return lambda x: x <= val
    elif s.startswith("<"):
        val = float(s[1:].strip())
        return lambda x: x < val
    elif "-" in s and not s.startswith("-"):
        parts = s.split("-")
        if len(parts) == 2:
            try:
                low, high = float(parts[0].strip()), float(parts[1].strip())
                return lambda x: low <= x <= high
            except ValueError:
                pass
    try:
        val = float(s)
        return lambda x: x == val
    except ValueError:
        return lambda x: str(x) == s


def _bin_series_by_list(series: pd.Series, bin_list: List[str]) -> pd.Series:
    """Bin numeric or parsed values of a Series into categorical buckets matching bin_list."""
    evaluators = [(b_str, _parse_bin_condition(b_str)) for b_str in bin_list]

    def _assign_bucket(val: Any) -> str:
        if pd.isna(val):
            return bin_list[0]
        for b_str, func in evaluators:
            try:
                if func(val):
                    return b_str
            except Exception:
                continue
        return str(val)

    binned = series.apply(_assign_bucket)
    return pd.Categorical(binned, categories=bin_list, ordered=True)


def _parse_and_prepare_data(
    df            : pd.DataFrame,
    category_col  : Optional[str],
    group_col     : Optional[str],
    bin_col       : Optional[str],
    bin_list      : Optional[List[str]],
    bin_func      : Optional[Any],
    cols_to_check : List[str],
    order         : Optional[List[str]],
    group_order   : Optional[List[str]]
) -> Tuple[pd.DataFrame, str, List[str], Optional[List[str]]]:
    """Validate columns, apply optional binning transformation, and determine plot orders."""
    if category_col is None and bin_col is None:
        raise ValueError("Either 'category_col' or 'bin_col' must be specified.")

    if bin_col is not None and bin_list is None and bin_func is None:
        raise ValueError("When 'bin_col' is specified, you must explicitly provide 'bin_list' or 'bin_func'.")

    resolved_cat_col = category_col if category_col is not None else "category_bin"

    required_cols = list(set(
        [c for c in [category_col, group_col, bin_col] + cols_to_check if c is not None]
    ))

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in dataframe: {missing}")

    plot_df = df[required_cols].copy()
    plot_df = plot_df.dropna(subset=[c for c in [category_col, group_col] if c is not None]).copy()

    if plot_df.empty:
        raise ValueError("No valid data remaining after dropping missing values.")

    if bin_col is not None:
        if callable(bin_func):
            raw_vals = plot_df[bin_col].apply(bin_func)
            binned_cats = pd.Categorical(raw_vals)
        else:
            raw_vals = plot_df[bin_col].apply(_count_items)
            binned_cats = _bin_series_by_list(raw_vals, bin_list)

        if category_col is None or category_col == "category_bin":
            plot_df[resolved_cat_col] = binned_cats
        else:
            plot_df[bin_col + "_binned"] = binned_cats

    if order is not None:
        final_cat_order = list(order)
    elif bin_list is not None and (category_col is None or category_col == "category_bin"):
        final_cat_order = list(bin_list)
    else:
        if isinstance(plot_df[resolved_cat_col].dtype, pd.CategoricalDtype):
            final_cat_order = [str(c) for c in plot_df[resolved_cat_col].cat.categories if c in plot_df[resolved_cat_col].values]
        else:
            final_cat_order = [str(c) for c in plot_df[resolved_cat_col].value_counts().index.tolist()]

    final_group_order = None
    if group_col is not None:
        if group_order is not None:
            final_group_order = list(group_order)
        else:
            final_group_order = sorted(plot_df[group_col].dropna().unique().tolist())

    return plot_df, resolved_cat_col, final_cat_order, final_group_order


def _determine_agg_func(col: Optional[str], op: str) -> str:
    """Determine underlying pandas aggregation function based on column name and operation."""
    if op in ("count", "nunique", "prop_count", "prop_nunique"):
        return "nunique" if op in ("nunique", "prop_nunique") else "count"
    if op in ("sum", "prop_sum"):
        return "sum"
    if op in ("mean", "median", "min", "max"):
        return op

    if col is None:
        return "count"
    col_lower = str(col).lower()
    if "size" in col_lower or "length" in col_lower or "count" in col_lower:
        return "sum"
    if any(k in col_lower for k in ("id", "reduced", "code", "name", "type", "key")):
        return "nunique"
    return "sum"


def _aggregate_panel_metric(
    plot_df      : pd.DataFrame,
    category_col : str,
    group_col    : Optional[str],
    col          : Optional[str],
    op           : str,
    cat_order    : List[str],
    group_order  : Optional[List[str]]
) -> pd.DataFrame:
    """Aggregate metric values and annotation counts per category and optional group."""
    if op not in AGG_OPS:
        raise ValueError(f"Unsupported operation '{op}'. Choose from {AGG_OPS}.")

    target_col = col if col is not None else category_col
    agg_func = _determine_agg_func(col, op)
    is_prop = op in ("proportion", "prop", "prop_sum", "prop_count", "prop_nunique")

    if group_col is not None:
        if group_order is None:
            group_order = sorted(plot_df[group_col].dropna().unique().tolist())

        full_index = pd.MultiIndex.from_product(
            [group_order, cat_order], names=[group_col, category_col]
        )

        df_res = (
            plot_df.groupby([group_col, category_col], observed=False)[target_col]
            .agg(agg_func)
            .reindex(full_index, fill_value=0)
            .reset_index(name="raw_val")
        )

        if is_prop:
            group_totals = df_res.groupby(group_col)["raw_val"].transform("sum")
            df_res["metric"] = np.where(group_totals > 0, df_res["raw_val"] / group_totals, 0.0)
        else:
            df_res["metric"] = df_res["raw_val"]

        df_res["annot"] = df_res["raw_val"]
        return df_res

    else:
        if col is None:
            counts = plot_df[category_col].value_counts().reindex(cat_order).fillna(0)
        else:
            counts = (
                plot_df.groupby(category_col, observed=False)[target_col]
                .agg(agg_func)
                .reindex(cat_order)
                .fillna(0)
            )

        if is_prop:
            total = float(counts.sum())
            props = counts / total if total > 0 else counts.astype(float)
            metric_vals = props.values
        else:
            metric_vals = counts.values

        return pd.DataFrame({
            category_col: cat_order,
            "metric": metric_vals,
            "annot": counts.values
        })


def _apply_panel_scale_and_limits(
    ax          : plt.Axes,
    df_metric   : pd.DataFrame,
    op          : str,
    scale       : str,
    orientation : str,
    panel_side  : str,
    xtick_step  : Optional[float] = None
) -> None:
    """Apply scale type (linear or log), tick formatting, and axis limits to a panel."""
    if scale not in ("linear", "log"):
        raise ValueError(f"Unsupported scale '{scale}'. Choose 'linear' or 'log'.")

    is_vert = (orientation == "vertical")
    is_prop = op in ("proportion", "prop", "prop_sum", "prop_count", "prop_nunique")

    if is_vert:
        ax.set_xscale(scale)
        lim_setter = ax.set_xlim
        ticks_getter = ax.get_xticks
        ticks_setter = ax.set_xticks
        ticklabels_setter = ax.set_xticklabels
    else:
        ax.set_yscale(scale)
        lim_setter = ax.set_ylim
        ticks_getter = ax.get_yticks
        ticks_setter = ax.set_yticks
        ticklabels_setter = ax.set_yticklabels

    max_val = df_metric["metric"].max()
    if pd.isna(max_val) or max_val <= 0:
        max_val = 1.0

    if scale == "linear":
        if is_prop:
            upper_lim = 1.05
            lim_setter(0, upper_lim) if panel_side == "primary" else lim_setter(upper_lim, 0)
            step = xtick_step if (xtick_step is not None and xtick_step > 0) else 0.2
            ticks = np.round(np.arange(0.0, 1.0 + step * 0.5, step), 4)
            ticks_setter(ticks)

            labels = []
            for t in ticks:
                if np.isclose(t, 0.0):
                    labels.append("0" if panel_side == "primary" else "")
                else:
                    labels.append(f"{t:g}")
            ticklabels_setter(labels)

        else:
            upper_lim = max(max_val * 1.15, 1.0)
            lim_setter(0, upper_lim) if panel_side == "primary" else lim_setter(upper_lim, 0)
            if xtick_step is not None and xtick_step > 0:
                ticks = np.round(np.arange(0.0, upper_lim + xtick_step * 0.5, xtick_step), 4)
                ticks_setter(ticks)

            ticks = ticks_getter()
            labels = []
            for t in ticks:
                if np.isclose(t, 0.0):
                    labels.append("0" if panel_side == "primary" else "")
                else:
                    if isinstance(t, (int, np.integer)) or (isinstance(t, float) and t.is_integer()):
                        labels.append(f"{int(t)}")
                    else:
                        labels.append(f"{t:g}")
            ticklabels_setter(labels)
    else:
        pos_vals = df_metric[df_metric["metric"] > 0]["metric"]
        min_pos = pos_vals.min() if not pos_vals.empty else (1e-1 if is_prop else 1.0)
        lower = min_pos * 0.5
        upper = max_val * 2.0
        if panel_side == "primary":
            lim_setter(lower, upper)
        else:
            lim_setter(upper, lower)


def _place_bar_annotation(
    ax          : plt.Axes,
    rect        : Any,
    val         : Any,
    orientation : str,
    panel_side  : str,
    fontsize    : float = 8
) -> None:
    """Position and render text annotation on an individual bar patch."""
    if val is None or pd.isna(val):
        return
    if val == 0 and rect.get_width() == 0 and rect.get_height() == 0:
        return

    if orientation == "vertical":
        pos = rect.get_width()
        coord = (pos, rect.get_y() + rect.get_height() / 2.0)
        offset = (3, 0) if panel_side == "primary" else (-3, 0)
        ha = "left" if panel_side == "primary" else "right"
        va = "center"
        rotation = 0
    else:
        pos = rect.get_height()
        coord = (rect.get_x() + rect.get_width() / 2.0, pos)
        offset = (0, 3) if panel_side == "primary" else (0, -3)
        ha = "center"
        va = "bottom" if panel_side == "primary" else "top"
        rotation = 90

    if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()):
        annot_text = f"{int(val)}"
    else:
        annot_text = f"{val:g}"

    ax.annotate(
        annot_text,
        xy=coord,
        xytext=offset,
        textcoords="offset points",
        ha=ha, va=va,
        fontsize=fontsize,
        rotation=rotation,
        color="black",
        fontweight="bold"
    )


def _annotate_panel_bars(
    ax           : plt.Axes,
    df_metric    : pd.DataFrame,
    category_col : str,
    cat_order    : List[str],
    plot_order   : List[str],
    group_col    : Optional[str],
    group_order  : Optional[List[str]],
    orientation  : str,
    panel_side   : str,
    fontsize     : float = 8
) -> None:
    """Annotate bar ends with bold values across primary or secondary panels."""
    if group_col is not None and group_order is not None:
        dict_map = df_metric.set_index([group_col, category_col])["annot"].to_dict()
        for group_idx, container in enumerate(ax.containers):
            if group_idx >= len(group_order):
                break
            group_val = group_order[group_idx]
            for cat_idx, rect in enumerate(container):
                if cat_idx >= len(plot_order):
                    break
                cat_val = plot_order[cat_idx]
                val = dict_map.get((group_val, cat_val), 0)
                _place_bar_annotation(ax, rect, val, orientation, panel_side, fontsize=fontsize)
    else:
        dict_map = df_metric.set_index(category_col)["annot"].to_dict()
        for i, cat_val in enumerate(plot_order):
            val = dict_map.get(cat_val, 0)
            if i < len(ax.patches):
                rect = ax.patches[i]
                _place_bar_annotation(ax, rect, val, orientation, panel_side, fontsize=fontsize)


def _setup_panel_legend(
    ax_prim         : plt.Axes,
    ax_sec          : plt.Axes,
    palette         : Dict[str, str],
    legend_order    : List[str],
    legend_title    : Optional[str],
    legend_fontsize : float,
    orientation     : str
) -> None:
    """Configure, render, or reposition legend on primary panel."""
    if ax_sec.get_legend() is not None:
        ax_sec.get_legend().remove()

    loc_pos = "center right" if orientation == "vertical" else "upper center"

    handles = [
        mpatches.Patch(facecolor=palette[k], label=k, edgecolor="black", linewidth=0.5)
        for k in legend_order if k in palette
    ]

    ax_prim.legend(
        handles=handles,
        title=legend_title,
        loc=loc_pos,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        frameon=True
    )


def plotBiPanelMetrics(
    df                   : pd.DataFrame,
    category_col         : Optional[str]            = None,
    group_col            : Optional[str]            = None,
    bin_col              : Optional[str]            = None,
    bin_list             : Optional[List[str]]      = None,
    bin_func             : Optional[Any]            = None,
    col_primary          : Optional[str]            = None,
    col_secondary        : Optional[str]            = None,
    op_primary           : str                      = "proportion",
    op_secondary         : str                      = "count",
    scale_primary        : str                      = "linear",
    scale_secondary      : str                      = "linear",
    orientation          : str                      = "vertical",
    swap                 : bool                     = False,
    invert_y_order       : Optional[bool]           = None,
    palette              : Optional[Dict[str, str]] = None,
    order                : Optional[List[str]]      = None,
    group_order          : Optional[List[str]]      = None,
    label_category       : Optional[str]            = None,
    label_primary        : str                      = "Primary Metric",
    label_secondary      : str                      = "Secondary Metric",
    title                : Optional[str]            = None,
    legend_title         : Optional[str]            = None,
    legend_fontsize      : float                    = 9,
    label_fontsize       : float                    = 11,
    bar_counts_fontsize  : float                    = 8.5,
    tick_rotation        : Optional[float]          = 0,
    show_counts          : bool                     = True,
    xtick_step_primary   : Optional[float]          = None,
    xtick_step_secondary : Optional[float]          = None,
    alpha_primary        : float                    = 1.0,
    alpha_secondary      : float                    = 1.0,
    ax                   : Optional[Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]] = None,
    figsize              : Tuple[float, float]      = (8, 6)
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot a bi-directional barplot sharing a single central baseline (vertical or horizontal):
      - Primary panel: Right panel (vertical layout) or Top panel (horizontal layout).
      - Secondary panel: Left panel (vertical layout) or Bottom panel (horizontal layout).
      - Single baseline origin labeled strictly as "0" on the primary panel.
    """
    sns.set_theme(context="paper", style="ticks")

    cols_to_check = [c for c in [col_primary, col_secondary] if c is not None]
    plot_df, resolved_cat_col, final_cat_order, final_group_order = _parse_and_prepare_data(
        df=df,
        category_col=category_col,
        group_col=group_col,
        bin_col=bin_col,
        bin_list=bin_list,
        bin_func=bin_func,
        cols_to_check=cols_to_check,
        order=order,
        group_order=group_order
    )

    color_keys = final_group_order if final_group_order is not None else final_cat_order
    if palette is None:
        colors = sns.color_palette("colorblind", n_colors=len(color_keys))
        palette = dict(zip(color_keys, colors))

    if not swap:
        c_prim, o_prim, s_prim, a_prim, lbl_prim, st_prim = (
            col_primary, op_primary, scale_primary, alpha_primary, label_primary, xtick_step_primary
        )
        c_sec, o_sec, s_sec, a_sec, lbl_sec, st_sec = (
            col_secondary, op_secondary, scale_secondary, alpha_secondary, label_secondary, xtick_step_secondary
        )
    else:
        c_prim, o_prim, s_prim, a_prim, lbl_prim, st_prim = (
            col_secondary, op_secondary, scale_secondary, alpha_secondary, label_secondary, xtick_step_secondary
        )
        c_sec, o_sec, s_sec, a_sec, lbl_sec, st_sec = (
            col_primary, op_primary, scale_primary, alpha_primary, label_primary, xtick_step_primary
        )

    df_prim = _aggregate_panel_metric(
        plot_df, resolved_cat_col, group_col, c_prim, o_prim, final_cat_order, final_group_order
    )
    df_sec = _aggregate_panel_metric(
        plot_df, resolved_cat_col, group_col, c_sec, o_sec, final_cat_order, final_group_order
    )

    if ax is None:
        if orientation == "vertical":
            fig, (ax_sec, ax_prim) = plt.subplots(
                1, 2, figsize=figsize, sharey=True, gridspec_kw={"width_ratios": [1, 1]}
            )
            fig.subplots_adjust(wspace=0.0)
        elif orientation == "horizontal":
            fig, (ax_prim, ax_sec) = plt.subplots(
                2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1, 1]}
            )
            fig.subplots_adjust(hspace=0.0)
        else:
            raise ValueError("Parameter 'orientation' must be 'vertical' or 'horizontal'.")
    elif isinstance(ax, (tuple, list, np.ndarray)) and len(ax) == 2:
        if orientation == "vertical":
            ax_sec, ax_prim = ax
        else:
            ax_prim, ax_sec = ax
    else:
        raise ValueError("Parameter 'ax' must be None or a two-element container of Axes.")

    rot = 0 if tick_rotation is None else tick_rotation
    hue_var = group_col if group_col is not None else resolved_cat_col
    hue_ord = final_group_order if group_col is not None else final_cat_order
    dodge_val = True if group_col is not None else False
    bar_width = 0.7 if group_col is not None else 0.35

    resolved_invert_y = (bin_col is not None) if invert_y_order is None else invert_y_order

    if resolved_invert_y:
        plot_cat_order = final_cat_order[::-1]
    else:
        plot_cat_order = final_cat_order

    if orientation == "vertical":
        sns.barplot(
            data=df_prim, y=resolved_cat_col, x="metric", hue=hue_var,
            hue_order=hue_ord, order=plot_cat_order, palette=palette,
            width=bar_width, edgecolor="black", linewidth=0.5, dodge=dodge_val,
            alpha=a_prim, ax=ax_prim
        )
        sns.barplot(
            data=df_sec, y=resolved_cat_col, x="metric", hue=hue_var,
            hue_order=hue_ord, order=plot_cat_order, palette=palette,
            width=bar_width, edgecolor="black", linewidth=0.5, dodge=dodge_val,
            alpha=a_sec, ax=ax_sec
        )
    else:
        sns.barplot(
            data=df_prim, x=resolved_cat_col, y="metric", hue=hue_var,
            hue_order=hue_ord, order=plot_cat_order, palette=palette,
            width=bar_width, edgecolor="black", linewidth=0.5, dodge=dodge_val,
            alpha=a_prim, ax=ax_prim
        )
        sns.barplot(
            data=df_sec, x=resolved_cat_col, y="metric", hue=hue_var,
            hue_order=hue_ord, order=plot_cat_order, palette=palette,
            width=bar_width, edgecolor="black", linewidth=0.5, dodge=dodge_val,
            alpha=a_sec, ax=ax_sec
        )

    _apply_panel_scale_and_limits(
        ax=ax_prim, df_metric=df_prim, op=o_prim, scale=s_prim,
        orientation=orientation, panel_side="primary", xtick_step=st_prim
    )
    _apply_panel_scale_and_limits(
        ax=ax_sec, df_metric=df_sec, op=o_sec, scale=s_sec,
        orientation=orientation, panel_side="secondary", xtick_step=st_sec
    )

    if show_counts:
        _annotate_panel_bars(
            ax=ax_prim, df_metric=df_prim, category_col=resolved_cat_col,
            cat_order=final_cat_order, plot_order=plot_cat_order,
            group_col=group_col, group_order=final_group_order,
            orientation=orientation, panel_side="primary", fontsize=bar_counts_fontsize
        )
        _annotate_panel_bars(
            ax=ax_sec, df_metric=df_sec, category_col=resolved_cat_col,
            cat_order=final_cat_order, plot_order=plot_cat_order,
            group_col=group_col, group_order=final_group_order,
            orientation=orientation, panel_side="secondary", fontsize=bar_counts_fontsize
        )

    _setup_panel_legend(
        ax_prim=ax_prim,
        ax_sec=ax_sec,
        palette=palette,
        legend_order=hue_ord,
        legend_title=legend_title,
        legend_fontsize=legend_fontsize,
        orientation=orientation
    )

    if orientation == "vertical":
        ax_prim.set_xlabel(lbl_prim, fontsize=label_fontsize, fontweight="bold", labelpad=10)
        ax_sec.set_xlabel(lbl_sec, fontsize=label_fontsize, fontweight="bold", labelpad=10)
        ax_prim.set_ylabel("")
        ax_prim.tick_params(axis="y", which="both", left=False, labelleft=False)

        if label_category is None:
            ax_sec.set_ylabel("")
            ax_sec.tick_params(axis="y", which="both", left=False, labelleft=False)
            if group_col is None:
                sns.despine(ax=ax_sec, left=True, right=True)
            else:
                sns.despine(ax=ax_sec, right=True)
        else:
            ax_sec.set_ylabel(label_category, fontsize=label_fontsize, fontweight="bold", labelpad=10)
            if rot != 0:
                plt.setp(ax_sec.get_yticklabels(), rotation=rot, ha="right", va="center")
            sns.despine(ax=ax_sec, right=True)

        sns.despine(ax=ax_prim, left=True)
        ax_prim.axvline(0 if s_prim == "linear" else ax_prim.get_xlim()[0],
                        color="black", linewidth=0.8, zorder=10, clip_on=False)
    else:
        ax_prim.set_ylabel(lbl_prim, fontsize=label_fontsize, fontweight="bold", labelpad=10)
        ax_sec.set_ylabel(lbl_sec, fontsize=label_fontsize, fontweight="bold", labelpad=10)
        ax_prim.set_xlabel("")
        ax_prim.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if label_category is None:
            ax_sec.set_xlabel("")
            ax_sec.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            sns.despine(ax=ax_sec, top=True, bottom=True)
        else:
            ax_sec.set_xlabel(label_category, fontsize=label_fontsize, fontweight="bold", labelpad=10)
            if rot != 0:
                plt.setp(ax_sec.get_xticklabels(), rotation=rot, ha="center", va="top")
            sns.despine(ax=ax_sec, top=False, bottom=False)

        sns.despine(ax=ax_prim, bottom=True)
        ax_sec.axhline(0 if s_sec == "linear" else ax_sec.get_ylim()[0],
                       color="black", linewidth=0.8, zorder=10, clip_on=False)

    if title:
        ax_prim.set_title(title, fontweight="bold", fontsize=11, pad=15)

    for sub_ax in (ax_sec, ax_prim):
        sub_ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        sub_ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        sub_ax.set_axisbelow(True)
        sub_ax.tick_params(axis="both", which="both", length=4, color="black")

    return ax_sec, ax_prim