import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators
from typing import Optional, Dict, List, Tuple


def prepareClusterIndicators(
    df              : pd.DataFrame,
    cluster_id_col  : str                      = "cluster_id",
    category_col    : Optional[str]            = None,
    value_cols_dict : Optional[Dict[str, str]] = None,
    categories      : Optional[List[str]]      = None,
    threshold       : float                    = 0.0
) -> pd.DataFrame:
    """
    Prepare a boolean indicator DataFrame aggregated at the cluster level.
    Supports both long-format DataFrames (e.g., ligand interactions per cluster)
    and wide-format DataFrames (e.g., method percentages per cluster).

    Parameters:
        * df :              a pandas DataFrame containing cluster data.
        * cluster_id_col :  a string. Column name for unique cluster identifiers.
                            Default is 'cluster_id'.
        * category_col :    a string or None. Column name containing categorical labels 
                            (for long-format DataFrames).
        * value_cols_dict : a dict or None. Mapping of category names to numerical 
                            column names (for wide-format DataFrames).
        * categories :      a list or None. Explicit order and selection of category names 
                            to include as columns.
        * threshold :       a float. Minimum value required to set boolean indicator to True. 
                            Default is 0.0.

    Returns:
        * pd.DataFrame : A DataFrame indexed by cluster_id with boolean indicator columns.
    """
    if cluster_id_col not in df.columns:
        raise KeyError(f"Missing expected cluster column in dataframe: '{cluster_id_col}'")

    if category_col is not None:
        if category_col not in df.columns:
            raise KeyError(f"Missing expected category column in dataframe: '{category_col}'")

        indicator_df = (
            df.groupby([cluster_id_col, category_col], observed=False)
            .size()
            .unstack(fill_value=0) > 0
        )

        if categories is not None:
            missing_cats = [c for c in categories if c not in indicator_df.columns]
            for mc in missing_cats:
                indicator_df[mc] = False
            indicator_df = indicator_df[categories]

        return indicator_df

    elif value_cols_dict is not None:
        indicator_df = pd.DataFrame(index=df[cluster_id_col])

        for cat_label, col_name in value_cols_dict.items():
            if col_name not in df.columns:
                raise KeyError(f"Missing expected column in dataframe: '{col_name}'")
            indicator_df[cat_label] = (df[col_name] > threshold).values

        if categories is not None:
            indicator_df = indicator_df[categories]

        return indicator_df

    else:
        raise ValueError("Either 'category_col' or 'value_cols_dict' must be provided.")


def plotClusterUpSet(
    indicator_df             : pd.DataFrame,
    categories               : List[str],
    palette                  : Optional[Dict[str, str]] = None,
    prioritize_singles       : bool                     = True,
    sort_singles_by          : str                      = "category_order",
    multi_intersection_color : str                      = "#000000",
    sort_by                  : Optional[str]            = "cardinality",
    show_counts              : bool                     = True,
    facecolor                : str                      = "#2b2b2b",
    orientation              : str                      = "horizontal",
    title                    : Optional[str]            = None,
    title_fontsize           : float                    = 14,
    fig                      : Optional[plt.Figure]     = None,
    figsize                  : Tuple[float, float]      = (9, 6)
) -> Dict[str, plt.Axes]:
    """
    Plot an UpSet diagram showing set overlaps across clusters with customizable 
    single-set ordering, specific color mapping for single sets, a unified color 
    for multi-group intersections, and orientation adjustments.

    Parameters:
        * indicator_df :             a pandas DataFrame indexed by cluster_id with boolean columns.
        * categories :               a list of strings. Ordered names of categories to include.
        * palette :                  a dict or None. Mapping of single category labels to HEX colors.
        * prioritize_singles :       a bool. If True, displays single-category sets first before 
                                     multi-category intersections. Default is True.
        * sort_singles_by :          a string. Order criteria for single-category sets when 
                                     prioritize_singles=True ('category_order' or 'cardinality'). 
                                     Default is 'category_order'.
        * multi_intersection_color : a string. HEX color used for all multi-group intersections 
                                     (> 1 group). Default is '#000000' (black).
        * sort_by :                  a string or None. Sorting rule for intersections if 
                                     prioritize_singles is False. Default is 'cardinality'.
        * show_counts :              a bool. If True, displays counts above/beside intersection bars.
        * facecolor :                a string. Default color for dots and unstyled bars.
        * orientation :              a string. Plot layout orientation ('horizontal' or 'vertical').
        * title :                    a string or None. Optional figure title. Supports '{n}' 
                                     placeholder for non-duplicated total cluster count.
        * title_fontsize :           a number. Font size for the title.
        * fig :                      a plt.Figure instance or None.
        * figsize :                  a tuple specifying figure dimensions.

    Returns:
        * Dict[str, plt.Axes] : A dictionary of Matplotlib Axes objects composing the UpSet plot.
    """
    for cat in categories:
        if cat not in indicator_df.columns:
            raise KeyError(f"Category '{cat}' not found in indicator DataFrame.")

    # ------------------------------------------------------------------
    # 1. AGGREGATE SUBSET COUNTS
    # ------------------------------------------------------------------
    counts = indicator_df.groupby(categories, observed=False).size()
    counts = counts[counts > 0]

    n_unique_clusters = int(indicator_df[categories].any(axis=1).sum())

    # ------------------------------------------------------------------
    # 2. CUSTOM SUBSET ORDERING
    # ------------------------------------------------------------------
    if prioritize_singles:
        single_tuples = []
        for cat in categories:
            tup = tuple(c == cat for c in categories)
            if tup in counts.index:
                single_tuples.append(tup)

        if sort_singles_by == "cardinality":
            single_tuples.sort(key=lambda idx: counts.loc[idx], reverse=True)
        elif sort_singles_by != "category_order":
            raise ValueError("Parameter 'sort_singles_by' must be 'category_order' or 'cardinality'.")

        remaining_tuples = [idx for idx in counts.index if idx not in single_tuples]
        remaining_tuples.sort(key=lambda idx: counts.loc[idx], reverse=True)

        custom_order = single_tuples + remaining_tuples
        counts = counts.reindex(custom_order)
        effective_sort_by = None
    else:
        effective_sort_by = sort_by

    # ------------------------------------------------------------------
    # 3. UPSET INITIALIZATION
    # ------------------------------------------------------------------
    upset = UpSet(
        counts,
        subset_size="sum",
        show_counts=show_counts,
        sort_by=effective_sort_by,
        orientation=orientation,
        facecolor=facecolor
    )

    # ------------------------------------------------------------------
    # 4. INTERSECTION COLORING (SINGLE SETS VS MULTI-GROUPS)
    # ------------------------------------------------------------------
    for idx in counts.index:
        present_cats = [cat for cat, is_present in zip(categories, idx) if is_present]
        absent_cats = [cat for cat, is_present in zip(categories, idx) if not is_present]

        if not present_cats:
            continue

        if len(present_cats) == 1:
            cat = present_cats[0]
            subset_color = palette.get(cat, facecolor) if palette is not None else facecolor
        else:
            subset_color = multi_intersection_color

        upset.style_subsets(
            present=present_cats,
            absent=absent_cats,
            facecolor=subset_color
        )

    # ------------------------------------------------------------------
    # 5. PLOTTING
    # ------------------------------------------------------------------
    if fig is None:
        fig = plt.figure(figsize=figsize)

    axes_dict = upset.plot(fig=fig)

    # ------------------------------------------------------------------
    # 6. VERTICAL ORIENTATION SPECIFIC ADJUSTMENTS
    # ------------------------------------------------------------------
    if orientation == "vertical":
        ax_inter = axes_dict.get("intersections")
        if ax_inter is not None:
            # Move intersection size X-axis to top and adjust label padding
            ax_inter.xaxis.tick_top()
            ax_inter.xaxis.set_label_position("top")
            ax_inter.xaxis.labelpad = 12

            # Remove bottom spine and display top spine
            ax_inter.spines["bottom"].set_visible(False)
            ax_inter.spines["top"].set_visible(True)

            # Separate show_counts labels slightly further from bar ends
            if show_counts:
                xlim = ax_inter.get_xlim()
                x_offset = (xlim[1] - xlim[0]) * 0.03
                for txt in ax_inter.texts:
                    x, y = txt.get_position()
                    txt.set_position((x + x_offset, y))
                ax_inter.set_xlim(xlim[0], xlim[1] + x_offset * 2)

        # Rotate category labels 90° (read from bottom to top)
        for ax_key in ("matrix", "totals"):
            sub_ax = axes_dict.get(ax_key)
            if sub_ax is not None:
                for lbl in sub_ax.get_xticklabels():
                    lbl.set_rotation(90)
                    lbl.set_ha("center")
                    lbl.set_va("bottom")

    # ------------------------------------------------------------------
    # 7. TITLE ANNOTATION
    # ------------------------------------------------------------------
    if title is not None:
        formatted_title = title.format(n=n_unique_clusters) if "{n}" in title else title
        fig.suptitle(formatted_title, fontweight="bold", fontsize=title_fontsize, y=1.03)

    return axes_dict