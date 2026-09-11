import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from typing import Optional, Dict, Tuple


def plotExperimentalMethodsVenn(
    df              : pd.DataFrame,
    cluster_id_col  : str                      = "cluster_id",
    col_xrd         : str                      = "perc_xrd",
    col_em          : str                      = "perc_em",
    col_nmr         : str                      = "perc_nmr",
    palette         : Optional[Dict[str, str]] = None,
    alpha           : float                    = 0.6,
    draw_circles    : bool                     = True,
    show_zeros      : bool                     = True,
    unweighted      : bool                     = True,
    title           : Optional[str]            = None,
    title_fontsize  : float                    = 12,
    label_fontsize  : float                    = 10,
    number_fontsize : float                    = 10,
    ax              : Optional[plt.Axes]       = None,
    figsize         : Tuple[float, float]      = (7, 7)
) -> plt.Axes:
    """
    Plot a 3-set Venn diagram showing the overlap of experimental methods across clusters.

    Parameters:
        * df :              a pandas DataFrame containing cluster experimental data.
        * cluster_id_col :  a string. Column name for unique cluster identifiers.
        * col_xrd :         a string. Column name for X-ray proportion. Default is 'perc_xrd'.
        * col_em :          a string. Column name for cryo-EM proportion. Default is 'perc_em'.
        * col_nmr :         a string. Column name for NMR proportion. Default is 'perc_nmr'.
        * palette :         a dictionary mapping method names to HEX colors.
        * alpha :           a float. Transparency level of the Venn sets (0.0 - 1.0).
        * draw_circles :    a bool. If True, draws outer circle outlines. Default is True.
        * show_zeros :      a bool. If True, explicitly displays '0' for empty intersections.
                            Default is True.
        * unweighted :      a bool. If True, forces equal-sized circle geometry so all 7 
                            intersections remain visible even if empty. Default is True.
        * title :           a string. Optional plot title.
        * title_fontsize :  a number. Font size for the title.
        * label_fontsize :  a number. Font size for set labels.
        * number_fontsize : a number. Font size for subset count numbers.
        * ax :              a plt.Axes instance or None.
        * figsize :         a tuple specifying figure dimensions.

    Returns:
        * plt.Axes : The matplotlib Axes object containing the Venn diagram.
    """
    for col in (cluster_id_col, col_xrd, col_em, col_nmr):
        if col not in df.columns:
            raise KeyError(f"Missing expected column in dataframe: '{col}'")

    # ------------------------------------------------------------------
    # 1. SETUP COLOR PALETTE & SETS
    # ------------------------------------------------------------------
    DEFAULT_METHOD_PALETTE = {
        'X-RAY DIFFRACTION'  : '#1b9e77', # Light Green
        'ELECTRON MICROSCOPY': '#d95f02', # Orange
        'SOLUTION NMR'       : '#7570b3', # Purple
    }
    color_map = DEFAULT_METHOD_PALETTE if palette is None else palette

    set_xrd = set(df.loc[df[col_xrd] > 0, cluster_id_col])
    set_em  = set(df.loc[df[col_em] > 0, cluster_id_col])
    set_nmr = set(df.loc[df[col_nmr] > 0, cluster_id_col])

    set_colors = (
        color_map.get('X-RAY DIFFRACTION', '#1b9e77'),
        color_map.get('ELECTRON MICROSCOPY', '#d95f02'),
        color_map.get('SOLUTION NMR', '#7570b3')
    )

    # ------------------------------------------------------------------
    # 2. CALCULATE 7 SUBSET COUNTS
    # ------------------------------------------------------------------
    counts = {
        '100': len(set_xrd - set_em - set_nmr),
        '010': len(set_em - set_xrd - set_nmr),
        '110': len((set_xrd & set_em) - set_nmr),
        '001': len(set_nmr - set_xrd - set_em),
        '101': len((set_xrd & set_nmr) - set_em),
        '011': len((set_em & set_nmr) - set_xrd),
        '111': len(set_xrd & set_em & set_nmr)
    }

    subsets_tuple = (
        counts['100'], counts['010'], counts['110'],
        counts['001'], counts['101'], counts['011'], counts['111']
    )

    # Use dummy equal sizes for layout if unweighted=True to preserve geometry
    draw_subsets = (1, 1, 1, 1, 1, 1, 1) if unweighted else subsets_tuple

    # ------------------------------------------------------------------
    # 3. AXES & VENN DRAWING
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    venn_diagram = venn3(
        subsets=draw_subsets,
        set_labels=('XRD', 'CEM', 'NMR'),
        set_colors=set_colors,
        alpha=alpha,
        ax=ax
    )

    if draw_circles:
        venn3_circles(
            subsets=draw_subsets,
            linestyle='solid',
            linewidth=1.0,
            color='black',
            ax=ax
        )

    # ------------------------------------------------------------------
    # 4. TEXT & FONT FORMATTING
    # ------------------------------------------------------------------
    if venn_diagram:
        for text in venn_diagram.set_labels:
            if text:
                text.set_fontsize(label_fontsize)
                text.set_fontweight('bold')

        for sid, val in counts.items():
            lbl = venn_diagram.get_label_by_id(sid)
            if lbl:
                lbl.set_fontsize(number_fontsize)
                if val == 0:
                    lbl.set_text('0' if show_zeros else '')
                else:
                    lbl.set_text(str(val))

    if title:
        ax.set_title(title, fontweight="bold", fontsize=title_fontsize, pad=15)

    return ax