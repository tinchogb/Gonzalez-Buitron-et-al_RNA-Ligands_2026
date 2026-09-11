"""
boxplot_comparison.py
=====================
Generalized module for comparative boxplots with statistical annotations
(Mann-Whitney U) and Bonferroni correction.

Supports:
  - Boxplots by main category (X-axis).
  - Optional sub-split by status (hue/colors).
  - Optional aggregated 'All' boxplot.
  - Independent palettes for categories and statuses.
  - Statistical annotations displayed as a separate legend below the color legend.
"""

import colorsys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.stats import mannwhitneyu


# =====================================================================
# 1. COLOR UTILITIES
# =====================================================================

def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert #RRGGBB to normalized RGB (0–1)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert normalized RGB to #RRGGBB."""
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255),
    )


def _adjust_color_opacity(hex_color: str, alpha: float = 0.6) -> str:
    """Blend a hex color with white to simulate opacity."""
    if alpha >= 1.0:
        return hex_color
    r, g, b = _hex_to_rgb(hex_color)
    r = r * alpha + (1 - alpha)
    g = g * alpha + (1 - alpha)
    b = b * alpha + (1 - alpha)
    return _rgb_to_hex(r, g, b)


# =====================================================================
# 2. DATA PROCESSING
# =====================================================================

def _prepare_dataframe(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    status_col: Optional[str] = None,
    show_all: bool = False,
    all_label: str = "All",
) -> pd.DataFrame:
    """
    Prepare the DataFrame for plotting.
    If show_all=True, append an 'All' row containing all data
    (status_col is preserved if present).
    """
    cols = [value_col, category_col]
    if status_col is not None:
        cols.append(status_col)

    plot_df = df[cols].copy()

    if show_all:
        all_df = plot_df.copy()
        all_df[category_col] = all_label
        plot_df = pd.concat([plot_df, all_df], ignore_index=True)

    return plot_df


def _resolve_orders(
    df: pd.DataFrame,
    category_col: str,
    status_col: Optional[str] = None,
    category_order: Optional[List[str]] = None,
    status_order: Optional[List[str]] = None,
    show_all: bool = False,
    all_label: str = "All",
) -> Tuple[List[str], Optional[List[str]]]:
    """Resolve the ordering of categorical variables."""
    if category_order is None:
        unique_cats = [c for c in df[category_col].unique() if c != all_label]
        cat_order = sorted(unique_cats, key=str)
    else:
        cat_order = list(category_order)

    if show_all and all_label not in cat_order:
        cat_order = cat_order + [all_label]

    if status_col is not None:
        if status_order is None:
            stat_order = sorted(df[status_col].dropna().unique(), key=str)
        else:
            stat_order = list(status_order)
    else:
        stat_order = None

    return cat_order, stat_order


def _resolve_palettes(
    df: pd.DataFrame,
    category_col: str,
    status_col: Optional[str] = None,
    palette_category: Optional[Dict[str, str]] = None,
    palette_status: Optional[Dict[str, str]] = None,
    show_all: bool = False,
    all_label: str = "All",
    all_color: str = "#DDDDDD",
    status_opacity_factor: float = 0.6,
) -> Tuple[Dict[str, str], Optional[Dict[str, str]], Dict[Tuple[str, Optional[str]], str]]:
    """
    Resolve color palettes.

    When palette_status is not provided, status colors are derived from
    the category palette by adjusting opacity (blending with white).

    Returns
    -------
    cat_palette : dict
        {category: hex_color} including All if applicable.
    status_palette : dict or None
        {status: hex_color} if provided.
    combined_colors : dict
        {(category, status): hex_color} for every visible patch.
    """
    # --- Category palette ---
    if palette_category is None:
        unique_cats = [c for c in df[category_col].unique() if c != all_label]
        n_cats = len(unique_cats)
        if n_cats > 0:
            default_colors = sns.color_palette("tab10", n_colors=n_cats).as_hex()
            cat_palette = {cat: default_colors[i] for i, cat in enumerate(unique_cats)}
        else:
            cat_palette = {}
    else:
        cat_palette = dict(palette_category)

    if show_all and all_label not in cat_palette:
        cat_palette[all_label] = all_color

    # --- Status palette and combined colors ---
    combined_colors: Dict[Tuple[str, Optional[str]], str] = {}

    if status_col is not None:
        unique_statuses = list(df[status_col].dropna().unique())

        if palette_status is not None:
            status_palette = dict(palette_status)
            for cat in cat_palette:
                for stat in unique_statuses:
                    combined_colors[(cat, stat)] = status_palette.get(stat, "#888888")
        else:
            status_palette = None
            for cat, base_color in cat_palette.items():
                for i, stat in enumerate(unique_statuses):
                    # First status: full opacity; subsequent: progressively lighter
                    alpha = 1.0 if i == 0 else (status_opacity_factor ** i)
                    combined_colors[(cat, stat)] = _adjust_color_opacity(
                        base_color, alpha=alpha
                    )
    else:
        status_palette = None
        for cat, color in cat_palette.items():
            combined_colors[(cat, None)] = color

    return cat_palette, status_palette, combined_colors


# =====================================================================
# 3. STATISTICAL MODULE
# =====================================================================

def _enumerate_boxplots(
    cat_order: List[str],
    stat_order: Optional[List[str]],
) -> List[Tuple[str, Optional[str]]]:
    """Enumerate all visible boxplots as (category, status) tuples."""
    if stat_order is not None:
        return [(cat, stat) for cat in cat_order for stat in stat_order]
    return [(cat, None) for cat in cat_order]


def _format_boxplot_name(
    cat: str, stat: Optional[str]
) -> str:
    """Format a boxplot identifier for display."""
    return f"{cat} - {stat}" if stat is not None else str(cat)


def _run_all_comparisons(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    status_col: Optional[str],
    boxplots: List[Tuple[str, Optional[str]]],
    bonferroni_factor: int,
) -> Dict[Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str]]], Tuple[float, str]]:
    """
    Run Mann-Whitney U for all C(M, 2) pairs of boxplots.

    Returns
    -------
    dict mapping (bp1, bp2) -> (raw_p_value, asterisks)
    """
    results: Dict = {}
    for bp1, bp2 in combinations(boxplots, 2):
        cat1, stat1 = bp1
        cat2, stat2 = bp2

        mask1 = df[category_col] == cat1
        mask2 = df[category_col] == cat2
        if status_col is not None:
            mask1 &= df[status_col] == stat1
            mask2 &= df[status_col] == stat2

        d1 = df.loc[mask1, value_col].dropna()
        d2 = df.loc[mask2, value_col].dropna()

        if len(d1) > 0 and len(d2) > 0:
            _, p = mannwhitneyu(d1, d2, alternative="two-sided")
            p_adj = min(1.0, p * max(1, bonferroni_factor))
            if p_adj < 0.001:
                ast = "***"
            elif p_adj < 0.01:
                ast = "**"
            elif p_adj < 0.05:
                ast = "*"
            else:
                ast = "ns"
            results[(bp1, bp2)] = (p, ast)
        else:
            results[(bp1, bp2)] = (1.0, "ns")
    return results


# =====================================================================
# 4. PLOT UTILITIES
# =====================================================================

def _patch_center_x(patch) -> float:
    """Return the center x-coordinate of a patch in data coordinates."""
    if isinstance(patch, mpatches.Rectangle):
        return patch.get_x() + patch.get_width() / 2.0
    elif isinstance(patch, mpatches.PathPatch):
        verts = patch.get_path().vertices
        return float(np.mean(verts[:, 0]))
    return 0.0


def _get_box_patches(ax: plt.Axes, expected_count: int) -> List:
    """
    Extract box-body patches from the axis in left-to-right order.

    Uses ax.patches (the list of patches added directly to the axis),
    excludes the background patch, and sorts by center-x.
    """
    candidates = []
    for p in ax.patches:
        if p is ax.patch:
            continue
        try:
            if isinstance(p, mpatches.Rectangle):
                if p.get_width() < 0.01:
                    continue
            cx = _patch_center_x(p)
            candidates.append((cx, p))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[0])
    return [p for _, p in candidates[:expected_count]]


def _apply_box_colors(
    ax: plt.Axes,
    cat_order: List[str],
    stat_order: Optional[List[str]],
    combined_colors: Dict[Tuple[str, Optional[str]], str],
) -> None:
    """Recolor boxplot patches based on their left-to-right position."""
    expected = len(cat_order) * (len(stat_order) if stat_order else 1)
    patches = _get_box_patches(ax, expected)

    idx = 0
    for cat in cat_order:
        for stat in (stat_order or [None]):
            if idx < len(patches):
                color = combined_colors.get((cat, stat), "#888888")
                patches[idx].set_facecolor(color)
                idx += 1


def _legend_bottom_y(fig, ax, legend) -> float:
    """
    Return the bottom y-coordinate (in axes fraction) of a legend.
    """
    try:
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        return float(bbox_axes.y0)
    except Exception:
        return 0.85


def _compute_group_whiskers(
    plot_df: pd.DataFrame,
    value_col: str,
    category_col: str,
    status_col: Optional[str],
    cat_order: List[str],
    stat_order: Optional[List[str]],
) -> Tuple[float, float]:
    """
    Compute the minimum lower-whisker and maximum upper-whisker
    across every visible group.

    Returns
    -------
    (min_whisker, max_whisker) : tuple of floats
    """
    min_whisker = float("inf")
    max_whisker = float("-inf")
    _tol = 1e-9  # tolerance for floating-point boundary comparisons

    for cat in cat_order:
        if status_col is not None and stat_order is not None:
            for stat in stat_order:
                series = plot_df[
                    (plot_df[category_col] == cat) &
                    (plot_df[status_col] == stat)
                ][value_col].dropna()
                if len(series) > 0:
                    q1, q3 = np.percentile(series, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr - _tol
                    upper_bound = q3 + 1.5 * iqr + _tol
                    lower = float(series[series >= lower_bound].min()) if len(series[series >= lower_bound]) > 0 else float(q1)
                    upper = float(series[series <= upper_bound].max()) if len(series[series <= upper_bound]) > 0 else float(q3)
                    min_whisker = min(min_whisker, lower)
                    max_whisker = max(max_whisker, upper)
        else:
            series = plot_df[plot_df[category_col] == cat][value_col].dropna()
            if len(series) > 0:
                q1, q3 = np.percentile(series, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr - _tol
                upper_bound = q3 + 1.5 * iqr + _tol
                lower = float(series[series >= lower_bound].min()) if len(series[series >= lower_bound]) > 0 else float(q1)
                upper = float(series[series <= upper_bound].max()) if len(series[series <= upper_bound]) > 0 else float(q3)
                min_whisker = min(min_whisker, lower)
                max_whisker = max(max_whisker, upper)

    if min_whisker == float("inf"):
        min_whisker = 0.0
    if max_whisker == float("-inf"):
        max_whisker = 1.0

    return min_whisker, max_whisker


def _snap_to_tick(value: float, tick_interval: float, direction: str = "down") -> float:
    """
    Snap a value to the nearest tick mark.
    direction='down' rounds toward negative infinity,
    direction='up' rounds toward positive infinity.
    """
    if direction == "down":
        return np.floor(value / tick_interval) * tick_interval
    else:
        return np.ceil(value / tick_interval) * tick_interval


# =====================================================================
# 5. MAIN PLOTTING FUNCTION
# =====================================================================

def plot_boxplot_comparison(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    status_col: Optional[str] = None,
    palette_category: Optional[Dict[str, str]] = None,
    palette_status: Optional[Dict[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend_title: Optional[str] = None,
    category_order: Optional[List[str]] = None,
    status_order: Optional[List[str]] = None,
    show_all: bool = False,
    all_label: str = "All",
    all_color: str = "#DDDDDD",
    use_log: bool = False,
    show_outliers: bool = True,
    status_opacity_factor: float = 0.6,
    y_tick_interval: float = 1.0,
    xtick_rotation: float = 0.0,
    xtick_size: float = 10.0,
    horizontal_alignment: str = "center",
    legend_title_fontsize: Optional[float] = None,
    legend_label_fontsize: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    add_stats_mann_whitney_u: bool = True,
    show_ns: bool = True,
    legend_mann_whitney_u: Optional[str] = "Mann-Whitney U",
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Axes:
    """
    Generate comparative boxplots with optional statistical annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    value_col : str
        Numeric column to plot on the Y-axis.
    category_col : str
        Categorical column for the X-axis.
    status_col : str, optional
        Additional categorical column for hue-based splitting.
    palette_category : dict, optional
        {category: hex_color} for main categories.
    palette_status : dict, optional
        {status: hex_color} for statuses. If not provided, colors are
        derived from the category palette by adjusting opacity.
    xlabel, ylabel, title : str, optional
        Axis labels and plot title. If None, nothing is displayed.
    legend_title : str, optional
        Title for the color legend. If None, no title is shown.
    category_order : list, optional
        Order of categories on the X-axis.
    status_order : list, optional
        Order of statuses in the legend.
    show_all : bool
        If True, append an 'All' category with pooled data.
    all_label : str
        Label for the aggregated category.
    all_color : str
        Color for the aggregated category (default: '#DDDDDD').
    use_log : bool
        Use logarithmic scale on the Y-axis.
    show_outliers : bool
        Display outlier fliers.
    status_opacity_factor : float
        Opacity factor for secondary statuses when palette_status is not
        provided. Values < 1.0 lighten the color by blending with white.
    y_tick_interval : float
        Interval between major ticks on the Y-axis (default: 1.0).
    xtick_rotation : float
        Rotation angle for x-axis tick labels (default: 0.0).
    xtick_size : float
        Font size for x-axis tick labels (default: 10.0).
    horizontal_alignment : str
        Horizontal alignment of x-axis tick labels (default: "center").
    legend_title_fontsize : float, optional
        Font size for legend titles.
    legend_label_fontsize : float, optional
        Font size for legend labels.
    ax : plt.Axes, optional
        Existing axis to draw on.
    add_stats_mann_whitney_u : bool
        Compute and display Mann-Whitney U comparisons as a legend.
    show_ns : bool
        Include non-significant comparisons in the stats legend.
    legend_mann_whitney_u : str, optional
        Title for the statistical legend. If None, no title is shown.
    figsize : tuple
        Figure size if a new axis is created.

    Returns
    -------
    plt.Axes
        The axis with the generated plot.
    """
    # --- Validations ---
    for col in [value_col, category_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    if status_col is not None and status_col not in df.columns:
        raise ValueError(f"Status column '{status_col}' not found in the DataFrame.")

    # --- Data preparation ---
    plot_df = _prepare_dataframe(
        df, value_col, category_col, status_col, show_all, all_label
    )
    cat_order, stat_order = _resolve_orders(
        plot_df, category_col, status_col,
        category_order, status_order, show_all, all_label
    )
    cat_palette, stat_palette, combined_colors = _resolve_palettes(
        plot_df, category_col, status_col,
        palette_category, palette_status,
        show_all, all_label, all_color, status_opacity_factor
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # =====================================================================
    # DRAW BOXPLOTS
    # =====================================================================
    legend_kw = {}
    if legend_title_fontsize is not None:
        legend_kw["title_fontsize"] = legend_title_fontsize
    if legend_label_fontsize is not None:
        legend_kw["fontsize"] = legend_label_fontsize

    if status_col is None:
        # Simple case: categories only
        sns.boxplot(
            data=plot_df,
            x=category_col,
            y=value_col,
            order=cat_order,
            palette=cat_palette,
            hue=category_col,
            legend=False,
            showfliers=show_outliers,
            ax=ax,
            width=0.6,
        )

        # Color legend
        color_patches = [
            Patch(
                facecolor=cat_palette.get(cat, "#888888"),
                edgecolor="black",
                label=cat,
            )
            for cat in cat_order
        ]
        leg_color = ax.legend(
            handles=color_patches,
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            ncol=1,
            **legend_kw,
        )

    else:
        # With statuses: use seaborn hue
        n_stats = len(stat_order) if stat_order else 1
        if stat_palette is not None:
            hue_palette = {
                stat: stat_palette.get(stat, "#888888")
                for stat in (stat_order or [])
            }
        else:
            hue_palette = {
                stat: f"C{i}" for i, stat in enumerate(stat_order or [])
            }

        sns.boxplot(
            data=plot_df,
            x=category_col,
            y=value_col,
            hue=status_col,
            order=cat_order,
            hue_order=stat_order,
            palette=hue_palette,
            showfliers=show_outliers,
            ax=ax,
            width=0.6,
            gap=0.08,
        )

        # Remove native seaborn legend (we replace it with a custom one)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Recolor patches manually
        _apply_box_colors(ax, cat_order, stat_order, combined_colors)

        # Custom color legend: <category> - <status>
        color_patches = []
        for cat in cat_order:
            for stat in (stat_order or [None]):
                color = combined_colors.get((cat, stat), "#888888")
                label = f"{cat} - {stat}" if stat is not None else str(cat)
                color_patches.append(
                    Patch(facecolor=color, edgecolor="black", label=label)
                )

        leg_color = ax.legend(
            handles=color_patches,
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            ncol=1,
            **legend_kw,
        )

    # =====================================================================
    # STATISTICAL LEGEND
    # =====================================================================
    if add_stats_mann_whitney_u:
        boxplots = _enumerate_boxplots(cat_order, stat_order)
        m = len(boxplots)
        total_comparisons = max(1, m * (m - 1) // 2)

        results = _run_all_comparisons(
            plot_df, value_col, category_col, status_col,
            boxplots, total_comparisons
        )

        # Sort by significance (*** first) then alphabetically by comparison
        priority = {"***": 0, "**": 1, "*": 2, "ns": 3}
        sorted_results = sorted(
            results.items(),
            key=lambda kv: (
                priority.get(kv[1][1], 4),
                _format_boxplot_name(*kv[0][0]),
                _format_boxplot_name(*kv[0][1]),
            ),
        )

        stat_labels = []
        max_name_len = max(
            len(_format_boxplot_name(*bp)) for (bp, _), (_, _) in sorted_results
        )
        for (bp1, bp2), (_, ast) in sorted_results:
            if ast == "ns" and not show_ns:
                continue
            name1 = _format_boxplot_name(*bp1)
            name2 = _format_boxplot_name(*bp2)
            label = f"{ast:>3} {name1:>{max_name_len}} vs {name2}"
            stat_labels.append(label)

        if stat_labels:
            invisible_patches = [
                Patch(facecolor="none", edgecolor="none", linewidth=0, label=lab)
                for lab in stat_labels
            ]

            # Dynamic positioning: place stats legend just below the color legend
            gap = 0.02
            y_bottom = _legend_bottom_y(fig, ax, leg_color)
            y_stats = max(0.05, y_bottom - gap)

            # Add the color legend back so it is not overwritten
            ax.add_artist(leg_color)

            stat_legend_kw = dict(legend_kw)
            stat_legend_kw["prop"] = {"family": "monospace"}
            if legend_label_fontsize is not None:
                stat_legend_kw["prop"]["size"] = legend_label_fontsize

            ax.legend(
                handles=invisible_patches,
                title=legend_mann_whitney_u,
                loc="upper left",
                bbox_to_anchor=(1.02, y_stats),
                frameon=True,
                ncol=1,
                handlelength=0,
                handletextpad=0,
                **stat_legend_kw,
            )

    # =====================================================================
    # AXIS CONFIGURATION
    # =====================================================================
    if use_log:
        ax.set_yscale("log")

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    else:
        ax.set_xlabel("")

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold")
    else:
        ax.set_ylabel("")

    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=11)

    # X-tick rotation
    if xtick_rotation != 0.0 or xtick_size != 10.0 or horizontal_alignment != "center":
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha=horizontal_alignment, fontsize=xtick_size)

    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # =====================================================================
    # Y-AXIS LIMITS
    # =====================================================================
    if not show_outliers:
        min_whisker, max_whisker = _compute_group_whiskers(
            plot_df, value_col, category_col, status_col,
            cat_order, stat_order
        )

        # Small margin so whisker caps are never clipped by the axis edge.
        # Fixed at 50% of the tick interval; never exceeds one full tick.
        margin = y_tick_interval * 0.50

        y_bottom = min_whisker - margin
        y_top = max_whisker + margin

        # y_bottom = _snap_to_tick(y_bottom, y_tick_interval, direction="down")
        y_top = _snap_to_tick(y_top, y_tick_interval, direction="up")

        # Do not force negative limits when all data is non-negative.
        # if y_bottom < 0 and min_whisker >= 0:
        #     y_bottom = 0.0

        ax.set_ylim(bottom=y_bottom, top=y_top)
        if not use_log:
            ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    else:
        if not use_log:
            ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))

    sns.despine(ax=ax)
    return ax