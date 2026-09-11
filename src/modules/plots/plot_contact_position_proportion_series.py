"""
plot_contact_position_proportion_series.py

Boxplot series of contact-position proportions across a range of persistence
thresholds, grouped by an arbitrary categorical variable (e.g., RNA type),
with optional lines connecting the medians of each group.

Two layout modes are supported:

    * "hue"     : a single Axes; one box per (threshold, group) combination,
                  groups are dodged and color-coded (direct comparison).
    * "separate": one Axes per group, sharing the y-axis limits (trellis-
                  style); every panel keeps its own ticks.

The proportion computation is shared with plotContactPositionProportion via
the internal helper _compute_contact_proportions().
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# Improved statistical engine (all C(M,2) pairs + Bonferroni + asterisks)
from modules.plots.boxplot_comparison import (
    _enumerate_boxplots,
    _run_all_comparisons,
)


# ----------------------------------------------------------------------
# INTERNAL HELPERS
# ----------------------------------------------------------------------
def _compute_contact_proportions(
    df: pd.DataFrame,
    cluster_col: str,
    group_col: str,
    interaction_suffix: str,
    min_holo_conformers: int,
    persistence_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute the per-cluster proportion of contact positions for a single
    interaction subset and a single persistence threshold.

    Parameters:

                * df :                    a pandas DataFrame. The cluster-level
                                          aggregated dataset (e.g., cluster_stats).
                * cluster_col :           a string. Column with cluster identifiers.
                * group_col :             a string. Column with the categorical
                                          grouping variable.
                * interaction_suffix :    a string. One of "all", "no_aid",
                                          "no_aid_no_metal".
                * min_holo_conformers :   an integer. Minimum number of
                                          ligand-associated (holo) conformers a
                                          cluster must possess to be included.
                * persistence_threshold : a float or None. If None, any position
                                          with at least one contact is counted
                                          (classic mode). If a float in [0, 1],
                                          only positions whose contact frequency
                                          among holo conformers meets the
                                          threshold are counted (persistence
                                          mode).

    Returns:

                * pd.DataFrame : One row per cluster, with columns
                  cluster_col, group_col, "total_positions",
                  "contact_positions" and "proportion".

    """
    allowed_suffixes = {"all", "no_aid", "no_aid_no_metal"}
    suffix_clean = interaction_suffix.strip().lower()
    if suffix_clean not in allowed_suffixes:
        raise ValueError(
            f"interaction_suffix must be one of {allowed_suffixes}, "
            f"got '{interaction_suffix}'."
        )

    interaction_col = f"nb_conformers_with_interactions_{suffix_clean}"
    holo_col = f"total_holos_{suffix_clean}"
    required_cols = [cluster_col, group_col, interaction_col, holo_col]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in dataframe: {missing}")

    plot_df = df[required_cols].dropna().copy()

    # Retain only clusters with >= min_holo_conformers holo conformers
    cluster_holo_counts = plot_df.groupby(cluster_col)[holo_col].max()
    valid_clusters = cluster_holo_counts[
        cluster_holo_counts >= min_holo_conformers
    ].index
    plot_df = plot_df[plot_df[cluster_col].isin(valid_clusters)].copy()

    if plot_df.empty:
        raise ValueError(
            f"No clusters found with at least {min_holo_conformers} holo "
            f"conformer(s) for suffix '{suffix_clean}'."
        )

    if persistence_threshold is None:
        # --- Classic mode: any contact counts ---
        total_pos = (
            plot_df.groupby(cluster_col)
            .size()
            .reset_index(name="total_positions")
        )
        has_contact = plot_df[interaction_col] > 0
        contact_pos = (
            plot_df[has_contact]
            .groupby(cluster_col)
            .size()
            .reset_index(name="contact_positions")
        )
    else:
        # --- Persistence mode: only holo positions, frequency >= threshold ---
        holo_mask = plot_df[holo_col] > 0
        holo_df = plot_df[holo_mask].copy()

        if holo_df.empty:
            raise ValueError(
                "No positions with ligand-associated conformers found "
                f"for suffix '{suffix_clean}'."
            )

        holo_df["_persistence"] = holo_df[interaction_col] / holo_df[holo_col]

        total_pos = (
            holo_df.groupby(cluster_col)
            .size()
            .reset_index(name="total_positions")
        )
        persistent_mask = holo_df["_persistence"] >= persistence_threshold
        contact_pos = (
            holo_df[persistent_mask]
            .groupby(cluster_col)
            .size()
            .reset_index(name="contact_positions")
        )

    prop_df = total_pos.merge(contact_pos, on=cluster_col, how="left")
    prop_df["contact_positions"] = prop_df["contact_positions"].fillna(0)
    prop_df["proportion"] = (
        prop_df["contact_positions"] / prop_df["total_positions"]
    )

    # Group label (assumed constant within a cluster; takes first occurrence)
    group_map = plot_df.groupby(cluster_col)[group_col].first().reset_index()
    prop_df = prop_df.merge(group_map, on=cluster_col, how="left")

    return prop_df


def _resolve_group_colors(
    group_order: List[str],
    palette: Optional[Dict[str, str]],
) -> Dict[str, Tuple[float, float, float]]:
    """Map each group to an RGB color tuple, honoring the user palette."""
    if palette is not None:
        missing = [g for g in group_order if g not in palette]
        if missing:
            raise KeyError(f"palette is missing colors for groups: {missing}")
        return {
            g: plt.matplotlib.colors.to_rgb(palette[g]) for g in group_order
        }
    default_palette = sns.color_palette(n_colors=max(len(group_order), 1))
    return {g: default_palette[i] for i, g in enumerate(group_order)}


def _box_label(bp) -> str:
    """Extract the raw display name from an engine boxplot tuple."""
    if isinstance(bp, (tuple, list)):
        return str(bp[0])
    return str(bp)


# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------
def plotContactPositionProportionSeries(
    df                       : pd.DataFrame,
    cluster_col              : str,
    group_col                : str,
    persistence_thresholds   : List[float],
    interaction_suffix       : str                      = "all",
    min_holo_conformers      : int                      = 2,
    mode                     : str                      = "hue",   # "hue" | "separate"
    palette                  : Optional[Dict[str, str]] = None,
    xlabel                   : Optional[str]            = "Persistence threshold",
    ylabel                   : Optional[str]            = "Proportion of contact positions",
    title                    : Optional[str]            = None,
    group_order              : Optional[List[str]]      = None,
    showfliers               : bool                     = True,
    show_medians_line        : bool                     = True,
    medians_line_kwargs      : Optional[Dict]           = None,
    connect_n_min            : int                      = 1,
    show_counts              : bool                     = False,
    threshold_fmt            : str                      = "{:.2f}",
    stats_mode               : Optional[str]            = None,
                                           # None | "within_group_vs_baseline"
                                           #      | "between_groups_per_threshold"
    show_stats_legend        : bool                     = True,
    show_ns                  : bool                     = True,
    legend_stats             : str                      = "Mann-Whitney U",
    show_group_legend        : bool                     = True,
    legend_group_title       : Optional[str]            = None,
    legend_in_panel          : Optional[int]            = None,
    legend_panel_loc         : str                      = "upper right",
    legend_fontsize          : Union[int, float, str]   = 9,
    legend_title_fontsize    : Union[int, float, str]   = 10,
    xlabel_fontsize          : Union[int, float, str]   = 11,
    ylabel_fontsize          : Union[int, float, str]   = 11,
    title_fontsize           : Union[int, float, str]   = 12,
    xlabel_pad               : float                    = 12,
    ylabel_pad               : float                    = 18,
    xtick_labelsize          : Union[int, float, str]   = 9,
    ytick_labelsize          : Union[int, float, str]   = 9,
    tick_length              : float                    = 4,
    show_group_annotation    : bool                     = False,
    legend_x_anchor          : float                    = 0.72,
    plot_right_margin        : float                    = 0.70,
    facet_ncol               : int                      = 2,
    y_pad                    : float                    = 0.05,
    return_results           : bool                     = False,
    ax                       : Optional[plt.Axes]       = None,
    figsize                  : Tuple[float, float]      = (10, 6),
) -> Union[plt.Axes, np.ndarray, Tuple[plt.Axes, Dict, pd.DataFrame]]:
    """
    Plot a series of boxplots showing how the proportion of sequence
    positions establishing at least one contact within a cluster decays
    as the contact-persistence threshold is increased, computed per
    cluster and grouped by an arbitrary categorical variable (e.g., RNA
    type, experimental method).

    Parameters:
        df: pandas DataFrame with cluster-level data.
        cluster_col: Column name for cluster identifiers.
        group_col: Column name for the grouping variable.
        persistence_thresholds: List of thresholds in [0, 1].
        interaction_suffix: "all", "no_aid", or "no_aid_no_metal".
        min_holo_conformers: Minimum holo conformers required per cluster.
        mode: "hue" or "separate".
        palette: Color palette dictionary mapped to groups.
        xlabel: Figure x-axis label.
        ylabel: Figure y-axis label.
        title: Optional figure suptitle.
        group_order: List of group order.
        showfliers: Whether to display outlier points.
        show_medians_line: Whether to draw connecting median lines.
        medians_line_kwargs: Kwargs for median connecting lines.
        connect_n_min: Minimum n required to draw connecting median line.
        show_counts: Whether to display sample sizes above boxes.
        threshold_fmt: Format string for threshold x-axis labels.
        stats_mode: Statistical comparison mode.
        show_stats_legend: Whether to display statistics legends.
        show_ns: Whether to display non-significant pairwise comparisons.
        legend_stats: Base title for statistical legends.
        show_group_legend: Whether to display group color legend.
        legend_group_title: Title of group legend. Default None.
        legend_in_panel: Optional 1-based index of panel to place legend inside.
        legend_panel_loc: Location string for in-panel legend (default "upper right").
        legend_fontsize: Font size for legend labels.
        legend_title_fontsize: Font size for legend titles.
        xlabel_fontsize: Font size for x-axis label.
        ylabel_fontsize: Font size for y-axis label.
        title_fontsize: Font size for title.
        xlabel_pad: Distance padding for the x-axis label from ticks.
        ylabel_pad: Distance padding for the y-axis label from ticks.
        xtick_labelsize: Font size for x-axis tick labels.
        ytick_labelsize: Font size for y-axis tick labels.
        tick_length: Length of axis tick marks.
        show_group_annotation: Whether to draw boxed group text inside panel.
        legend_x_anchor: Figure x-coordinate for legend placement (when legend_in_panel is None).
        plot_right_margin: Right subplots adjustment margin.
        facet_ncol: Number of columns in "separate" mode.
        y_pad: Upper padding fraction for y-axis.
        return_results: Whether to return statistical results and DataFrame.
        ax: Matplotlib Axes (valid only in "hue" mode).
        figsize: Figure dimensions tuple.

    Returns:
        plt.Axes or np.ndarray of Axes, and optionally (axes, results, series_df).
    """
    # ------------------------------------------------------------------
    # 1. VALIDATION OF ARGUMENTS
    # ------------------------------------------------------------------
    mode_clean = mode.strip().lower()
    if mode_clean not in {"hue", "separate"}:
        raise ValueError(
            f"mode must be one of {{'hue', 'separate'}}, got '{mode}'."
        )

    if stats_mode is not None:
        allowed_stats = {
            "within_group_vs_baseline",
            "between_groups_per_threshold",
        }
        stats_clean = stats_mode.strip().lower()
        if stats_clean not in allowed_stats:
            raise ValueError(
                f"stats_mode must be one of {allowed_stats} or None, "
                f"got '{stats_mode}'."
            )
        if stats_clean == "between_groups_per_threshold" and mode_clean != "hue":
            raise ValueError(
                "stats_mode='between_groups_per_threshold' requires mode='hue' "
                "because it compares groups that live on different panels in "
                "'separate' mode."
            )
    else:
        stats_clean = None

    if not persistence_thresholds:
        raise ValueError("persistence_thresholds must be a non-empty list.")

    thresholds = sorted(set(float(t) for t in persistence_thresholds))
    out_of_range = [t for t in thresholds if not (0.0 <= t <= 1.0)]
    if out_of_range:
        raise ValueError(
            f"All persistence thresholds must lie in [0, 1], "
            f"got: {out_of_range}."
        )

    threshold_labels = [threshold_fmt.format(t) for t in thresholds]
    if len(set(threshold_labels)) != len(threshold_labels):
        raise ValueError(
            f"threshold_fmt='{threshold_fmt}' produces duplicate x-axis "
            f"labels for thresholds {thresholds}. Use a finer format."
        )

    if connect_n_min < 1:
        raise ValueError(f"connect_n_min must be >= 1, got {connect_n_min}.")

    # ------------------------------------------------------------------
    # 2. COMPUTE PROPORTIONS ACROSS THE THRESHOLD SWEEP
    # ------------------------------------------------------------------
    frames = []
    for t in thresholds:
        prop_df = _compute_contact_proportions(
            df=df,
            cluster_col=cluster_col,
            group_col=group_col,
            interaction_suffix=interaction_suffix,
            min_holo_conformers=min_holo_conformers,
            persistence_threshold=t,
        )
        prop_df = prop_df[[cluster_col, group_col, "proportion"]].copy()
        prop_df["threshold"] = t
        prop_df["threshold_label"] = threshold_fmt.format(t)
        frames.append(prop_df)

    series_df = pd.concat(frames, ignore_index=True)
    series_df["threshold_label"] = pd.Categorical(
        series_df["threshold_label"],
        categories=threshold_labels,
        ordered=True,
    )

    df_groups = set(df[group_col].dropna().unique())
    if group_order is None:
        group_order = sorted(df_groups)
    else:
        group_order = list(group_order)
        unknown = [g for g in group_order if g not in df_groups]
        if unknown:
            raise KeyError(
                f"group_order contains groups not present in "
                f"'{group_col}': {unknown}"
            )

    series_df = series_df[series_df[group_col].isin(group_order)].copy()
    group_colors = _resolve_group_colors(group_order, palette)

    # Compute N (unique clusters) per group
    group_counts = (
        series_df.groupby(group_col)[cluster_col]
        .nunique()
        .to_dict()
    )
    max_n_len = max(
        (len(str(group_counts.get(g, 0))) for g in group_order), default=1
    )

    default_line_kwargs = {
        "linewidth": 1.5,
        "marker": "o",
        "markersize": 5,
        "markeredgecolor": "black",
        "markeredgewidth": 0.8,
        "zorder": 5,
        "alpha": 0.95,
    }
    line_kwargs = {**default_line_kwargs, **(medians_line_kwargs or {})}

    width = 0.6
    n_levels = len(group_order)
    results: Dict = {}

    LEGEND_GAP = 0.015

    # Automatic margin expansion when placing legend inside panel
    effective_right_margin = (
        0.98 if legend_in_panel is not None and plot_right_margin == 0.70
        else plot_right_margin
    )

    def _render_and_center_legends(
        target_ax: plt.Axes,
        legend_items: List[Tuple[Optional[str], list, bool]],
        x_anchor: float,
        y_center: float,
        in_panel: bool = False,
    ) -> None:
        """Render legend stack either outside centered or inside target panel."""
        if not legend_items:
            return

        figure = target_ax.figure

        if in_panel:
            for leg_title, handles, monospace in legend_items:
                kwargs = dict(
                    handles=handles,
                    title=leg_title,
                    loc=legend_panel_loc,
                    frameon=True,
                    ncol=1,
                    title_fontsize=legend_title_fontsize,
                    prop={"family": "monospace", "size": legend_fontsize},
                )
                leg = target_ax.legend(**kwargs)
                target_ax.add_artist(leg)
            return

        measured_heights = []
        temp_legends = []

        # 1. Pre-render legends to measure height
        for leg_title, handles, monospace in legend_items:
            kwargs = dict(
                handles=handles,
                title=leg_title,
                loc="upper left",
                bbox_to_anchor=(x_anchor, 0.5),
                bbox_transform=figure.transFigure,
                frameon=True,
                ncol=1,
                title_fontsize=legend_title_fontsize,
                prop={"family": "monospace", "size": legend_fontsize},
            )
            leg = target_ax.legend(**kwargs)
            temp_legends.append(leg)

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()

        for leg in temp_legends:
            bb = leg.get_window_extent(renderer)
            height_frac = bb.height / figure.bbox.height
            measured_heights.append(height_frac)
            leg.remove()

        # 2. Compute stack height and top starting position
        total_stack_height = sum(measured_heights) + (len(legend_items) - 1) * LEGEND_GAP
        y_top = y_center + (total_stack_height / 2.0)
        if y_top > 0.98:
            y_top = 0.98

        # 3. Final placement from top to bottom
        y_cursor = y_top
        for (leg_title, handles, monospace), h_frac in zip(legend_items, measured_heights):
            kwargs = dict(
                handles=handles,
                title=leg_title,
                loc="upper left",
                bbox_to_anchor=(x_anchor, y_cursor),
                bbox_transform=figure.transFigure,
                frameon=True,
                ncol=1,
                title_fontsize=legend_title_fontsize,
                prop={"family": "monospace", "size": legend_fontsize},
            )

            leg = target_ax.legend(**kwargs)
            target_ax.add_artist(leg)
            try:
                leg.set_in_layout(True)
            except AttributeError:
                pass

            if leg not in figure.legends:
                figure.legends.append(leg)

            y_cursor -= (h_frac + LEGEND_GAP)

    def _stats_labels(pairs: Dict) -> List[str]:
        """Format comparison results as monospace labels."""
        priority = {"***": 0, "**": 1, "*": 2, "ns": 3}
        items = sorted(
            pairs.items(),
            key=lambda kv: (
                priority.get(kv[1][1], 4),
                _box_label(kv[0][0]),
                _box_label(kv[0][1]),
            ),
        )
        if not items:
            return []
        max_len = max(
            max(len(_box_label(bp1)), len(_box_label(bp2)))
            for (bp1, bp2), _ in items
        )
        labels = []
        for (bp1, bp2), (_, ast) in items:
            if ast == "ns" and not show_ns:
                continue
            labels.append(
                f"{ast:>3} {_box_label(bp1):>{max_len}} vs {_box_label(bp2)}"
            )
        return labels

    def _baseline_pairs(pairs: Dict, baseline: str) -> Dict:
        """Keep only comparisons involving the baseline threshold."""
        return {
            k: v for k, v in pairs.items()
            if baseline in (_box_label(k[0]), _box_label(k[1]))
        }

    def _group_handles() -> list:
        """Patch handles for group legend formatted as (n = <val>) - <category>."""
        handles = []
        for g in group_order:
            n_val = group_counts.get(g, 0)
            n_str = f"{n_val:>{max_n_len}d}"
            label = f"(n = {n_str}) - {g}"
            handles.append(
                Patch(facecolor=group_colors[g], edgecolor="black", label=label)
            )
        return handles

    def _draw_median_line(
        panel: plt.Axes, group: str, x_positions: Dict[str, float]
    ) -> None:
        """Draw median-connecting line for one group."""
        if not show_medians_line:
            return
        g_df = series_df[series_df[group_col] == group]
        xs, ys = [], []
        for lab in threshold_labels:
            sub = g_df[g_df["threshold_label"] == lab]["proportion"]
            if sub.empty or sub.size < connect_n_min:
                continue
            xs.append(x_positions[lab])
            ys.append(float(sub.median()))
        if not xs:
            return
        panel.plot(xs, ys, color=group_colors[group], **line_kwargs)

    def _draw_counts(
        panel: plt.Axes, group: str, x_positions: Dict[str, float]
    ) -> None:
        """Annotate sample sizes above Q3."""
        if not show_counts:
            return
        g_df = series_df[series_df[group_col] == group]
        for lab in threshold_labels:
            sub = g_df[g_df["threshold_label"] == lab]["proportion"]
            if sub.empty:
                continue
            q3 = float(np.percentile(sub, 75))
            panel.text(
                x_positions[lab] + 0.03, q3 + 0.01, f"{sub.size}",
                ha="left", va="bottom",
                fontsize=8, color="#333333", fontweight="bold",
            )

    def _hue_offsets() -> Dict[str, float]:
        """Dodge offset of each group within threshold."""
        return {
            g: width * (i - (n_levels - 1) / 2) / n_levels
            for i, g in enumerate(group_order)
        }

    # ------------------------------------------------------------------
    # 3a. MODE "HUE"
    # ------------------------------------------------------------------
    if mode_clean == "hue":
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            ax.figure.subplots_adjust(
                right=effective_right_margin, bottom=0.14
            )

        sns.boxplot(
            data=series_df,
            x="threshold_label",
            y="proportion",
            order=threshold_labels,
            hue=group_col,
            hue_order=group_order,
            palette=palette,
            legend=False,
            showfliers=showfliers,
            width=width,
            ax=ax,
        )

        offsets = _hue_offsets()
        for g in group_order:
            x_pos = {
                lab: i + offsets[g] for i, lab in enumerate(threshold_labels)
            }
            _draw_median_line(ax, g, x_pos)
            _draw_counts(ax, g, x_pos)

        legend_entries: List[Tuple[str, List[str]]] = []

        if stats_clean == "within_group_vs_baseline":
            baseline = threshold_labels[0]
            for g in group_order:
                g_df = series_df[series_df[group_col] == g]
                if g_df.empty:
                    results[g] = {}
                    continue
                boxplots = _enumerate_boxplots(threshold_labels, None)
                m = len(boxplots)
                pairs = _run_all_comparisons(
                    df=g_df,
                    value_col="proportion",
                    category_col="threshold_label",
                    status_col=None,
                    boxplots=boxplots,
                    bonferroni_factor=max(1, m * (m - 1) // 2),
                )
                results[g] = pairs
                labels = _stats_labels(_baseline_pairs(pairs, baseline))
                if labels:
                    legend_entries.append((f"{legend_stats} — {g}", labels))

        elif stats_clean == "between_groups_per_threshold":
            for lab in threshold_labels:
                t_df = series_df[series_df["threshold_label"] == lab]
                boxplots = _enumerate_boxplots(group_order, None)
                m = len(boxplots)
                pairs = _run_all_comparisons(
                    df=t_df,
                    value_col="proportion",
                    category_col=group_col,
                    status_col=None,
                    boxplots=boxplots,
                    bonferroni_factor=max(1, m * (m - 1) // 2),
                )
                results[lab] = pairs
                labels = [
                    f"[{lab}] {line}"
                    for line in _stats_labels(pairs)
                ]
                if labels:
                    legend_entries.append((f"{legend_stats} — {lab}", labels))

        # Collect legend items
        legend_items: List[Tuple[Optional[str], list, bool]] = []
        if show_group_legend:
            legend_items.append((legend_group_title, _group_handles(), True))
        if show_stats_legend and legend_entries:
            for entry_title, labels in legend_entries:
                handles = [
                    Patch(facecolor="none", edgecolor="none", linewidth=0, label=lab)
                    for lab in labels
                ]
                legend_items.append((entry_title, handles, True))

        # Render legends
        pos = ax.get_position()
        y_center_plot = (pos.y0 + pos.y1) / 2.0
        _render_and_center_legends(
            target_ax=ax,
            legend_items=legend_items,
            x_anchor=legend_x_anchor,
            y_center=y_center_plot,
            in_panel=(legend_in_panel is not None),
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=xtick_labelsize)
        plt.setp(ax.get_yticklabels(), fontsize=ytick_labelsize)

        if xlabel:
            ax.set_xlabel(
                xlabel, fontweight="bold", labelpad=xlabel_pad, fontsize=xlabel_fontsize
            )
        if ylabel:
            ax.set_ylabel(
                ylabel, fontweight="bold", labelpad=ylabel_pad, fontsize=ylabel_fontsize
            )
        if title:
            ax.set_title(title, fontweight="bold", fontsize=title_fontsize)

        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        max_y = float(series_df["proportion"].max())
        ax.set_ylim(bottom=-0.05, top=max_y + y_pad * max(max_y, 1e-9) + 0.02)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        ax.tick_params(
            axis="x", which="both", bottom=True, length=tick_length, color="black", labelsize=xtick_labelsize
        )
        ax.tick_params(
            axis="y", which="both", left=True, length=tick_length, color="black", labelsize=ytick_labelsize
        )

        sns.despine(ax=ax)
        axes_out: Union[plt.Axes, np.ndarray] = ax

    # ------------------------------------------------------------------
    # 3b. MODE "SEPARATE"
    # ------------------------------------------------------------------
    else:
        if ax is not None:
            raise ValueError(
                "The 'ax' argument is only valid in mode='hue'."
            )

        n_panels = len(group_order)
        ncols = min(facet_ncol, n_panels)
        nrows = math.ceil(n_panels / ncols)
        fig, axes_arr = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharey=False,
            squeeze=False,
        )
        axes_flat = axes_arr.ravel()
        legend_entries: List[Tuple[str, List[str]]] = []

        for idx, g in enumerate(group_order):
            panel = axes_flat[idx]
            g_df = series_df[series_df[group_col] == g]

            if not g_df.empty:
                sns.boxplot(
                    data=g_df,
                    x="threshold_label",
                    y="proportion",
                    order=threshold_labels,
                    hue=None,
                    color=group_colors[g],
                    legend=False,
                    showfliers=showfliers,
                    width=width * 0.7,
                    ax=panel,
                )
                x_pos = {
                    lab: float(i) for i, lab in enumerate(threshold_labels)
                }
                _draw_median_line(panel, g, x_pos)
                _draw_counts(panel, g, x_pos)
            else:
                panel.text(
                    0.5, 0.5, "No data",
                    transform=panel.transAxes, ha="center", va="center",
                    fontsize=10, color="#888888",
                )
                panel.set_xticks(range(len(threshold_labels)))
                panel.set_xticklabels(threshold_labels)

            panel.set_xlabel("")
            panel.set_ylabel("")

            if show_group_annotation:
                panel.text(
                    0.04, 0.96, str(g),
                    transform=panel.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="white",
                        edgecolor="#333333",
                        linewidth=0.8,
                    ),
                )

            panel.tick_params(
                axis="x", which="both", bottom=True, length=tick_length, color="black", labelsize=xtick_labelsize
            )
            panel.tick_params(
                axis="y", which="both", left=True, length=tick_length, color="black", labelsize=ytick_labelsize
            )
            panel.xaxis.grid(True, linestyle="--", alpha=0.5)
            panel.yaxis.grid(True, linestyle="--", alpha=0.5)
            panel.yaxis.set_major_locator(MultipleLocator(0.1))
            plt.setp(
                panel.get_xticklabels(), rotation=45, ha="right", fontsize=xtick_labelsize
            )
            plt.setp(
                panel.get_yticklabels(), fontsize=ytick_labelsize
            )
            sns.despine(ax=panel)

            if stats_clean == "within_group_vs_baseline":
                baseline = threshold_labels[0]
                if g_df.empty:
                    results[g] = {}
                else:
                    boxplots = _enumerate_boxplots(threshold_labels, None)
                    m = len(boxplots)
                    pairs = _run_all_comparisons(
                        df=g_df,
                        value_col="proportion",
                        category_col="threshold_label",
                        status_col=None,
                        boxplots=boxplots,
                        bonferroni_factor=max(1, m * (m - 1) // 2),
                    )
                    results[g] = pairs
                    labels = _stats_labels(_baseline_pairs(pairs, baseline))
                    if labels:
                        legend_entries.append(
                            (f"{legend_stats} — {g}", labels)
                        )

        for idx in range(n_panels, axes_flat.size):
            axes_flat[idx].set_visible(False)

        if not series_df.empty:
            max_y = float(series_df["proportion"].max())
            top = max_y + y_pad * max(max_y, 1e-9) + 0.02
            for panel in axes_flat[:n_panels]:
                panel.set_ylim(bottom=-0.05, top=top)

        if title:
            fig.suptitle(title, fontweight="bold", fontsize=title_fontsize)

        fig.tight_layout(
            rect=(0.02, 0.05, effective_right_margin, 0.97 if title else 1.0)
        )

        # Measure bounding box of active subplots for centering
        fig.canvas.draw()
        visible_axes = axes_flat[:n_panels]
        x0_min = min(a.get_position().x0 for a in visible_axes)
        x1_max = max(a.get_position().x1 for a in visible_axes)
        y0_min = min(a.get_position().y0 for a in visible_axes)
        y1_max = max(a.get_position().y1 for a in visible_axes)

        x_center = (x0_min + x1_max) / 2.0
        y_center = (y0_min + y1_max) / 2.0

        fig_w, fig_h = fig.get_size_inches()
        xlabel_pad_frac = (xlabel_pad / 72.0) / fig_h
        ylabel_pad_frac = (ylabel_pad / 72.0) / fig_w

        supxlabel_y = max(0.01, y0_min - xlabel_pad_frac - (0.02 * (xlabel_fontsize / 11.0)))
        supylabel_x = max(0.005, x0_min - ylabel_pad_frac - (0.02 * (ylabel_fontsize / 11.0)))

        if xlabel:
            fig.supxlabel(
                xlabel,
                x=x_center,
                y=supxlabel_y,
                fontweight="bold",
                fontsize=xlabel_fontsize,
                ha="center",
                va="top",
            )
        if ylabel:
            fig.supylabel(
                ylabel,
                x=supylabel_x,
                y=y_center,
                fontweight="bold",
                fontsize=ylabel_fontsize,
                ha="right",
                va="center",
            )

        # Collect legend items
        legend_items: List[Tuple[Optional[str], list, bool]] = []
        if show_group_legend:
            legend_items.append((legend_group_title, _group_handles(), True))
        if show_stats_legend and legend_entries:
            for entry_title, labels in legend_entries:
                handles = [
                    Patch(facecolor="none", edgecolor="none", linewidth=0, label=lab)
                    for lab in labels
                ]
                legend_items.append((entry_title, handles, True))

        # Determine target Axes for legend placement
        if legend_in_panel is not None:
            if not (1 <= legend_in_panel <= n_panels):
                raise ValueError(
                    f"legend_in_panel must be between 1 and {n_panels}, "
                    f"got {legend_in_panel}."
                )
            target_legend_ax = axes_flat[legend_in_panel - 1]
            in_panel_flag = True
        else:
            target_legend_ax = axes_flat[0]
            in_panel_flag = False

        _render_and_center_legends(
            target_ax=target_legend_ax,
            legend_items=legend_items,
            x_anchor=legend_x_anchor,
            y_center=y_center,
            in_panel=in_panel_flag,
        )

        axes_out = axes_flat[:n_panels]

    if return_results:
        return axes_out, results, series_df
    return axes_out