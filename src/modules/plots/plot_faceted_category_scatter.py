"""
plot_faceted_category_scatter.py

Grid scatterplots stratified by category with independent axis ticks,
shared axis scales, side-aligned figure legends or in-panel legends,
statistical metrics, and optional cluster ring highlighting with arrow annotations.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def _compute_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    method: str = "spearman",
) -> Tuple[Optional[float], Optional[float], str, str]:
    """Calculate correlation coefficient, p-value, and formatted labels."""
    clean_data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean_data) < 2:
        return None, None, "ns", "N/A"

    try:
        if method.lower() == "pearson":
            r, p = pearsonr(clean_data["x"], clean_data["y"])
            symbol = "Pearson r"
        else:
            r, p = spearmanr(clean_data["x"], clean_data["y"])
            symbol = "Spearman ρ"
    except Exception:
        return None, None, "ns", "N/A"

    if p < 0.001:
        p_str = "***"
        p_fmt = "p < 0.001"
    elif p < 0.01:
        p_str = "**"
        p_fmt = f"p = {p:.3f}"
    elif p < 0.05:
        p_str = "*"
        p_fmt = f"p = {p:.3f}"
    else:
        p_str = "ns"
        p_fmt = f"p = {p:.3f}"

    panel_label = f"{symbol} = {r:.2f}, {p_fmt} {p_str}"
    return float(r), float(p), p_str, panel_label


def _resolve_category_colors(
    category_order: List[str],
    palette: Optional[Dict[str, str]],
) -> Dict[str, Tuple[float, float, float]]:
    """Map each category to an RGB color tuple."""
    if palette is not None:
        missing = [c for c in category_order if c not in palette]
        if missing:
            raise KeyError(f"palette is missing colors for categories: {missing}")
        return {c: plt.matplotlib.colors.to_rgb(palette[c]) for c in category_order}

    default_palette = sns.color_palette(n_colors=max(len(category_order), 1))
    return {c: default_palette[i] for i, c in enumerate(category_order)}


def _format_category_legend_labels(
    clean_df: pd.DataFrame,
    category_col: str,
    category_order: List[str],
) -> List[Tuple[str, str]]:
    """Format category labels with padded sample sizes: '(n = <value>) - <category>'."""
    counts = clean_df[category_col].value_counts().to_dict()
    max_n_len = max(len(str(counts.get(cat, 0))) for cat in category_order)

    formatted_items = []
    for cat in category_order:
        n_val = counts.get(cat, 0)
        label_str = f"(n = {n_val:>{max_n_len}}) - {cat}"
        formatted_items.append((cat, label_str))

    return formatted_items


def _render_and_center_legends(
    target_ax: plt.Axes,
    legend_items: List[Tuple[Optional[str], list, bool, bool]],
    x_anchor: float,
    y_center: float,
    legend_fontsize: Union[int, float],
    legend_title_fontsize: Union[int, float],
) -> None:
    """
    Render a vertical stack of legends centered at y_center on the figure.
    legend_items tuple structure: (title, handles, is_monospace, hide_handles)
    """
    if not legend_items:
        return

    figure = target_ax.figure
    measured_heights = []
    temp_legends = []
    legend_gap = 0.025

    # 1. Measure legend heights
    for leg_title, handles, is_monospace, hide_handles in legend_items:
        kwargs = dict(
            handles=handles,
            title=leg_title,
            loc="upper left",
            bbox_to_anchor=(x_anchor, 0.5),
            bbox_transform=figure.transFigure,
            frameon=True,
            ncol=1,
            title_fontsize=legend_title_fontsize,
        )

        prop_dict = {"size": legend_fontsize}
        if is_monospace:
            prop_dict["family"] = "monospace"

        if hide_handles:
            kwargs.update(handlelength=0, handletextpad=0)

        kwargs["prop"] = prop_dict
        leg = target_ax.legend(**kwargs)
        temp_legends.append(leg)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for leg in temp_legends:
        bb = leg.get_window_extent(renderer)
        height_frac = bb.height / figure.bbox.height
        measured_heights.append(height_frac)
        leg.remove()

    # 2. Compute stack vertical layout
    total_stack_height = sum(measured_heights) + (len(legend_items) - 1) * legend_gap
    y_top = y_center + (total_stack_height / 2.0)
    if y_top > 0.98:
        y_top = 0.98

    # 3. Final placement
    y_cursor = y_top
    for (leg_title, handles, is_monospace, hide_handles), h_frac in zip(legend_items, measured_heights):
        kwargs = dict(
            handles=handles,
            title=leg_title,
            loc="upper left",
            bbox_to_anchor=(x_anchor, y_cursor),
            bbox_transform=figure.transFigure,
            frameon=True,
            ncol=1,
            title_fontsize=legend_title_fontsize,
        )

        prop_dict = {"size": legend_fontsize}
        if is_monospace:
            prop_dict["family"] = "monospace"

        if hide_handles:
            kwargs.update(handlelength=0, handletextpad=0)

        kwargs["prop"] = prop_dict
        leg = target_ax.legend(**kwargs)
        target_ax.add_artist(leg)

        if leg not in figure.legends:
            figure.legends.append(leg)

        y_cursor -= (h_frac + legend_gap)


def _render_in_panel_legends(
    target_ax: plt.Axes,
    legend_items: List[Tuple[Optional[str], list, bool, bool]],
    loc: str = "upper right",
    legend_fontsize: Union[int, float] = 9,
    legend_title_fontsize: Union[int, float] = 10,
) -> None:
    """Render legend items stacked inside a specific subplot panel Axes."""
    if not legend_items:
        return

    if len(legend_items) == 1:
        leg_title, handles, is_monospace, hide_handles = legend_items[0]
        kwargs = dict(
            handles=handles,
            title=leg_title,
            loc=loc,
            frameon=True,
            ncol=1,
            title_fontsize=legend_title_fontsize,
        )
        prop_dict = {"size": legend_fontsize}
        if is_monospace:
            prop_dict["family"] = "monospace"

        if hide_handles:
            kwargs.update(handlelength=0, handletextpad=0)

        kwargs["prop"] = prop_dict
        target_ax.legend(**kwargs)
        return

    figure = target_ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    is_lower = "lower" in loc.lower()
    is_left = "left" in loc.lower()
    is_center_x = "center" in loc.lower() and not ("upper" in loc.lower() or "lower" in loc.lower())

    x_anchor = 0.04 if is_left else (0.5 if is_center_x else 0.96)
    y_anchor = 0.04 if is_lower else 0.96
    anchor_loc = loc if ("upper" in loc.lower() or "lower" in loc.lower()) else ("lower right" if is_lower else "upper right")

    legend_gap = 0.02
    temp_legends = []
    measured_heights = []

    for leg_title, handles, is_monospace, hide_handles in legend_items:
        kwargs = dict(
            handles=handles,
            title=leg_title,
            loc=anchor_loc,
            bbox_to_anchor=(x_anchor, y_anchor),
            bbox_transform=target_ax.transAxes,
            frameon=True,
            ncol=1,
            title_fontsize=legend_title_fontsize,
        )
        prop_dict = {"size": legend_fontsize}
        if is_monospace:
            prop_dict["family"] = "monospace"

        if hide_handles:
            kwargs.update(handlelength=0, handletextpad=0)

        kwargs["prop"] = prop_dict
        leg = target_ax.legend(**kwargs)
        temp_legends.append(leg)

    figure.canvas.draw()
    for leg in temp_legends:
        bb = leg.get_window_extent(renderer)
        inv = target_ax.transAxes.inverted()
        bb_axes = inv.transform_bbox(bb)
        measured_heights.append(bb_axes.height)
        leg.remove()

    y_cursor = y_anchor
    for idx, ((leg_title, handles, is_monospace, hide_handles), h_frac) in enumerate(zip(legend_items, measured_heights)):
        curr_loc = "lower left" if (is_lower and is_left) else ("lower right" if is_lower else ("upper left" if is_left else "upper right"))
        kwargs = dict(
            handles=handles,
            title=leg_title,
            loc=curr_loc,
            bbox_to_anchor=(x_anchor, y_cursor),
            bbox_transform=target_ax.transAxes,
            frameon=True,
            ncol=1,
            title_fontsize=legend_title_fontsize,
        )
        prop_dict = {"size": legend_fontsize}
        if is_monospace:
            prop_dict["family"] = "monospace"

        if hide_handles:
            kwargs.update(handlelength=0, handletextpad=0)

        kwargs["prop"] = prop_dict
        leg = target_ax.legend(**kwargs)
        if idx < len(legend_items) - 1:
            target_ax.add_artist(leg)

        if is_lower:
            y_cursor += h_frac + legend_gap
        else:
            y_cursor -= (h_frac + legend_gap)


# ----------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------

def plotFacetedCategoryScatter(
    df                     : pd.DataFrame,
    x_col                  : str,
    y_col                  : str,
    category_col           : str,
    category_order         : Optional[List[str]]          = None,
    highlight_clusters     : Optional[List[Union[int, str]]] = None,
    cluster_id_col         : str                          = "cluster_id",
    palette                : Optional[Dict[str, str]]     = None,
    xlabel                 : Optional[str]                = None,
    ylabel                 : Optional[str]                = None,
    title                  : Optional[str]                = None,
    show_legend            : bool                         = True,
    legend_title           : Optional[str]                = None,
    legend_in_panel        : Optional[int]                = None,
    legend_panel_loc       : str                          = "upper right",
    show_scatter           : bool                         = True,
    show_regression_line   : bool                         = True,
    confidence_interval    : Optional[int]                = 95,
    show_category_badge    : bool                         = False,
    show_correlation       : bool                         = True,
    correlation_method     : str                          = "spearman", # "spearman" | "pearson"
    correlation_display    : str                          = "both",     # "panel" | "legend" | "both"
    legend_stats_title     : Optional[str]                = None,
    alpha                  : float                        = 0.5,
    s                      : float                        = 25,
    share_scales           : bool                         = True,
    facet_ncol             : int                          = 4,
    figsize                : Tuple[float, float]          = (16, 4),
    legend_x_anchor        : float                        = 0.84,
    plot_right_margin      : Optional[float]              = None,
    xlabel_pad             : float                        = 12,
    ylabel_pad             : float                        = 18,
    xlabel_fontsize        : Union[int, float]            = 11,
    ylabel_fontsize        : Union[int, float]            = 11,
    title_fontsize         : Union[int, float]            = 12,
    legend_fontsize        : Union[int, float]            = 9,
    legend_title_fontsize  : Union[int, float]            = 10,
    return_results         : bool                         = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Dict[str, float]], pd.DataFrame]]:
    """
    Plot a grid of category-stratified scatterplots with independent panel ticks,
    shared axis scales, side-aligned figure legends or in-panel legends, statistical
    metrics, and optional cluster ring highlighting with arrow annotations.

    Parameters:
        df: pandas DataFrame containing plotting columns.
        x_col: Column name for x-axis continuous variable.
        y_col: Column name for y-axis continuous variable.
        category_col: Column name defining both category faceting and color.
        category_order: Explicit category order list.
        highlight_clusters: List of cluster IDs to highlight with a black ring and arrow label.
        cluster_id_col: Column name containing cluster identifiers. Default is "cluster_id".
        palette: Dictionary mapping category names to colors.
        xlabel: Figure-level x-axis label.
        ylabel: Figure-level y-axis label.
        title: Global figure title (suptitle).
        show_legend: Whether to display the global category color legend.
        legend_title: Title for the category legend box. If None, no title is displayed.
        legend_in_panel: Index of the panel inside which legends should be placed.
                         Supports 0-based or 1-based indexing. If None, legends are placed
                         outside on the right side of the figure.
        legend_panel_loc: Location string for placing legends inside a panel when
                          legend_in_panel is specified (e.g., "upper right", "upper left").
                          Default is "upper right".
        show_scatter: Whether to draw scatter points.
        show_regression_line: Whether to overlay linear regression fit lines.
        confidence_interval: Size of confidence interval for regression (0-100).
                             If None, no confidence band is drawn. Default is 95.
        show_category_badge: Whether to display a category name box inside subplots.
                             Default is False.
        show_correlation: Whether to calculate statistical correlation metrics.
        correlation_method: "spearman" or "pearson".
        correlation_display: "panel", "legend", or "both".
        legend_stats_title: Custom title for correlation legend block.
        alpha: Point transparency (0 to 1).
        s: Marker point size.
        share_scales: Whether all subplots share identical x and y limits.
        facet_ncol: Number of grid columns for subplots.
        figsize: Figure dimensions tuple (width, height).
        legend_x_anchor: Figure X-coordinate for placing right-hand side legends.
        plot_right_margin: Subplot right margin boundary adjustment. Default is 0.81 if
                           outer legend is active, or 0.98 if legend_in_panel is specified.
        xlabel_pad: Padding for x-axis label.
        ylabel_pad: Padding for y-axis label.
        xlabel_fontsize: Font size for x-axis label.
        ylabel_fontsize: Font size for y-axis label.
        title_fontsize: Font size for figure title.
        legend_fontsize: Font size for legend entries.
        legend_title_fontsize: Font size for legend section titles.
        return_results: Whether to return axes array, correlation stats dict, and plot df.

    Returns:
        np.ndarray of Axes, or (axes, stats_dict, clean_df) if return_results=True.
    """
    # ------------------------------------------------------------------
    # 1. VALIDATION & DATA PREPARATION
    # ------------------------------------------------------------------
    required_cols = [x_col, y_col, category_col]
    if highlight_clusters is not None and cluster_id_col in df.columns:
        required_cols.append(cluster_id_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    clean_df = df[required_cols].dropna(subset=[x_col, y_col, category_col]).copy()
    if clean_df.empty:
        raise ValueError("No valid data remaining after dropping missing values.")

    available_categories = clean_df[category_col].unique()
    if category_order is None:
        category_order = sorted(available_categories)
    else:
        category_order = [c for c in category_order if c in available_categories]

    if not category_order:
        raise ValueError("No matching categories found in dataset for plotting.")

    category_colors = _resolve_category_colors(category_order, palette)
    corr_display_clean = correlation_display.strip().lower()

    if plot_right_margin is None:
        effective_right_margin = 0.98 if legend_in_panel is not None else 0.81
    else:
        effective_right_margin = plot_right_margin

    # ------------------------------------------------------------------
    # 2. GRID INITIALISATION
    # ------------------------------------------------------------------
    n_panels = len(category_order)
    ncols = min(facet_ncol, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes_arr = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes_arr.ravel()
    fig.subplots_adjust(right=effective_right_margin)

    stats_results: Dict[str, Dict[str, float]] = {}
    corr_legend_labels: List[str] = []

    # Calculate global scale ranges if share_scales=True
    x_min, x_max = clean_df[x_col].min(), clean_df[x_col].max()
    y_min, y_max = clean_df[y_col].min(), clean_df[y_col].max()
    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1

    # ------------------------------------------------------------------
    # 3. PANEL RENDERING
    # ------------------------------------------------------------------
    for idx, cat in enumerate(category_order):
        panel = axes_flat[idx]
        sub_df = clean_df[clean_df[category_col] == cat]
        color = category_colors[cat]

        # 3a. Scatter plot
        if show_scatter:
            sns.scatterplot(
                data=sub_df,
                x=x_col,
                y=y_col,
                color=color,
                alpha=alpha,
                s=s,
                edgecolor="none",
                legend=False,
                ax=panel,
            )

        # 3b. Regression line & Confidence Interval
        if show_regression_line and len(sub_df) > 2:
            sns.regplot(
                data=sub_df,
                x=x_col,
                y=y_col,
                scatter=False,
                ci=confidence_interval,
                color=color,
                line_kws={"linewidth": 1.5},
                ax=panel,
            )

        # 3c. Highlight Clusters with Black Ring & Arrow Line
        if highlight_clusters is not None and cluster_id_col in sub_df.columns:
            highlight_df = sub_df[sub_df[cluster_id_col].isin(highlight_clusters)]
            if not highlight_df.empty:
                # Outer black ring
                panel.scatter(
                    highlight_df[x_col],
                    highlight_df[y_col],
                    s=s * 2.5,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=6,
                )

                # Offset for text position relative to data range
                x_span = (x_max - x_min) if x_max != x_min else 1.0
                y_span = (y_max - y_min) if y_max != y_min else 1.0
                x_offset = x_span * 0.15
                y_offset = y_span * 0.05

                for _, row_data in highlight_df.iterrows():
                    cid = row_data[cluster_id_col]
                    px, py = row_data[x_col], row_data[y_col]

                    panel.annotate(
                        str(cid),
                        xy=(px, py),
                        xytext=(px + x_offset, py + y_offset),
                        fontsize=8,
                        fontweight="bold",
                        color="#111111",
                        ha="left",
                        va="bottom",
                        zorder=7,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.8,
                        ),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            facecolor="black",
                            edgecolor="black",
                            linewidth=1.0,
                            mutation_scale=10,
                            shrinkA=4, # Space at text label end
                            shrinkB=5, # Space at scatter point end
                        ),
                    )

        # 3d. Optional Panel Category Badge (Upper Right Box)
        if show_category_badge:
            panel.text(
                0.96, 0.96, str(cat),
                transform=panel.transAxes,
                ha="right", va="top",
                fontsize=10, fontweight="bold", color="#333333",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.9,
                ),
            )

        # 3e. Correlation Metrics
        if show_correlation:
            r_val, p_val, p_str, panel_label = _compute_correlation_stats(
                sub_df[x_col], sub_df[y_col], method=correlation_method
            )
            if r_val is not None:
                stats_results[cat] = {"r": r_val, "p_value": p_val}

                # Panel correlation text box (Upper Left)
                if corr_display_clean in ("panel", "both"):
                    panel.text(
                        0.04, 0.96, panel_label,
                        transform=panel.transAxes,
                        ha="left", va="top",
                        fontsize=9, fontweight="bold", color="#333333",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="gray",
                            alpha=0.9,
                        ),
                    )

                # Collect legend correlation entry
                symbol = "ρ" if correlation_method.lower() == "spearman" else "r"
                max_cat_len = max(len(str(c)) for c in category_order)
                corr_legend_labels.append(
                    f"{str(cat):>{max_cat_len}}: {symbol} = {r_val:.2f} ({p_str})"
                )

        # 3f. Styling & Independent Ticks
        panel.set_xlabel("")
        panel.set_ylabel("")

        if share_scales:
            panel.set_xlim(x_min - x_margin, x_max + x_margin)
            panel.set_ylim(y_min - y_margin, y_max + y_margin)

        panel.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
            labelbottom=True,
            labelleft=True,
            length=4,
            color="black",
        )
        panel.xaxis.grid(True, linestyle="--", alpha=0.5)
        panel.yaxis.grid(True, linestyle="--", alpha=0.5)
        sns.despine(ax=panel)

    # Hide unused grid panels
    for idx in range(n_panels, axes_flat.size):
        axes_flat[idx].set_visible(False)

    visible_axes = axes_flat[:n_panels]

    # ------------------------------------------------------------------
    # 4. SUBPLOT GRID BOUNDING BOX & CENTERED LABELS
    # ------------------------------------------------------------------
    fig.canvas.draw()
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

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=title_fontsize)

    # ------------------------------------------------------------------
    # 5. CONSOLIDATED LEGENDS (FIGURE-LEVEL OR IN-PANEL)
    # ------------------------------------------------------------------
    legend_items: List[Tuple[Optional[str], list, bool, bool]] = []

    # Category color legend with aligned (n = <value>) labels
    if show_legend:
        cat_labels = _format_category_legend_labels(clean_df, category_col, category_order)
        group_handles = [
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=category_colors[cat],
                markersize=max(s * 0.35, 6),
                label=label_str,
            )
            for cat, label_str in cat_labels
        ]
        legend_items.append((
            legend_title,
            group_handles,
            True,  # is_monospace
            False, # hide_handles (keep marker dots visible)
        ))

    # Correlation statistics legend
    if show_correlation and corr_display_clean in ("legend", "both") and corr_legend_labels:
        stats_title = (
            legend_stats_title
            if legend_stats_title is not None
            else f"{correlation_method.capitalize()} Test"
        )
        stats_handles = [
            Patch(facecolor="none", edgecolor="none", linewidth=0, label=lab)
            for lab in corr_legend_labels
        ]
        legend_items.append((
            stats_title,
            stats_handles,
            True, # is_monospace
            True, # hide_handles (remove extra empty indentation)
        ))

    if legend_in_panel is not None:
        if 0 <= legend_in_panel < len(visible_axes):
            panel_idx = legend_in_panel
        elif 1 <= legend_in_panel <= len(visible_axes):
            panel_idx = legend_in_panel - 1
        else:
            raise IndexError(
                f"legend_in_panel={legend_in_panel} is out of bounds for "
                f"{len(visible_axes)} panel(s)."
            )
        _render_in_panel_legends(
            target_ax=visible_axes[panel_idx],
            legend_items=legend_items,
            loc=legend_panel_loc,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
        )
    else:
        # Render stacked legends centered vertically against the active subplot grid
        _render_and_center_legends(
            target_ax=visible_axes[0],
            legend_items=legend_items,
            x_anchor=legend_x_anchor,
            y_center=y_center,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
        )

    if return_results:
        return visible_axes, stats_results, clean_df
    return visible_axes