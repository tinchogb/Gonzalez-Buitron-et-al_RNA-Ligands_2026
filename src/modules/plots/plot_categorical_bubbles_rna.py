from typing import Literal, Optional, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


def plotCategoricalBubblesRNA(df                       : pd.DataFrame
                            , palette                  : dict
                            , primaryCol               : str  = 'rna_type_cdrna'
                            , secondaryCol             : str  = 'max_rmsd'
                            , sizeCol                  : str  = 'cluster_size'
                            , primaryLabel             : Optional[str] = 'RNA Type'
                            , secondaryLabel           : Optional[str] = 'Maximum RMSD (Å)'
                            , title                    : Optional[str] = 'RNA Type vs Maximum RMSD'
                            , legend_title             : str  = 'cluster size'
                            , order_primary            : Optional[list] = None
                            , orientation              : Literal['vertical', 'horizontal'] = 'vertical'
                            , figsize                  : Tuple[float, float] = (7, 7)
                            , primary_label_fontsize   : float = 12.0
                            , secondary_label_fontsize : float = 12.0
                            , legend_fontsize          : float = 10.0
                            , title_fontsize           : float = 14.0 ) -> plt.Axes:
    """Plots primary categorical variable against secondary numerical variable with bubble sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing primary categories, secondary values, and cluster sizes.
    palette : dict
        Color palette mapping primary category types to hex colors.
    primaryCol : str, default='rna_type_cdrna'
        Column name for primary categorical variable.
    secondaryCol : str, default='max_rmsd'
        Column name for secondary continuous variable.
    sizeCol : str, default='cluster_size'
        Column name for cluster sizes determining scatter point dimensions.
    primaryLabel : str or None, default='RNA Type'
        Label for the primary axis. Set to None to omit.
    secondaryLabel : str or None, default='Maximum RMSD (Å)'
        Label for the secondary axis. Set to None to omit.
    title : str or None, default='RNA Type vs Maximum RMSD'
        Title for the plot. Set to None to omit.
    legend_title : str, default='cluster size'
        Title for the size legend.
    order_primary : list or None, default=None
        Custom category ordering list for the primary variable.
    orientation : {'vertical', 'horizontal'}, default='vertical'
        Layout orientation. 'vertical' places primary on X axis (left to right),
        'horizontal' on Y axis (top to bottom).
    figsize : tuple of float, default=(7, 7)
        Size of the figure in inches (width, height).
    primary_label_fontsize : float, default=12.0
        Font size for the primary axis label.
    secondary_label_fontsize : float, default=12.0
        Font size for the secondary axis label.
    legend_fontsize : float, default=10.0
        Font size for legend entries and legend title.
    title_fontsize : float, default=14.0
        Font size for plot title.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object containing the plot.
    """
    if orientation not in ('vertical', 'horizontal'):
        raise ValueError("orientation must be either 'vertical' or 'horizontal'")

    working_df = df.copy()
    working_df[primaryCol] = working_df[primaryCol].fillna('unknown').replace('', 'unknown')
    
    # Identify categories actually present in the data
    present_types = set(working_df[primaryCol].unique())

    # 1. Canonical ordering logic
    if order_primary is not None:
        sorted_types = [t for t in order_primary if t in present_types]
        if 'unknown' in present_types and 'unknown' not in sorted_types:
            sorted_types.append('unknown')
    else:
        stats = working_df.groupby(primaryCol)[secondaryCol].max().sort_values(ascending=False)
        sorted_types = [t for t in stats.index if t != 'unknown']
        if 'unknown' in present_types:
            sorted_types.append('unknown')
        
    # Vertical -> left to right | Horizontal -> top to bottom (reverse list for Y-axis)
    plot_categories = sorted_types if orientation == 'vertical' else sorted_types[::-1]
    working_df[primaryCol] = pd.Categorical(working_df[primaryCol], categories=plot_categories, ordered=True)
    
    # Native constrained layout handles external top/right legend spacing dynamically
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    
    # User palette configuration
    plot_palette = palette.copy()
    if 'unknown' not in plot_palette:
        plot_palette['unknown'] = '#000000'

    # 2. Map axes according to orientation
    if orientation == 'vertical':
        x_var, y_var = primaryCol, secondaryCol
    else:
        x_var, y_var = secondaryCol, primaryCol

    # 3. Scatterplot size scaling
    min_s, max_s = 10, 396
    sizes_range = (min_s, max_s)
    
    scatter = sns.scatterplot(
        data=working_df, x=x_var, y=y_var, size=sizeCol, hue=primaryCol,
        palette=plot_palette, sizes=sizes_range, alpha=0.6, edgecolor='white', ax=ax,
        legend=False,
    )
    
    # 4. Legend positioning and handle ordering conditional on plot orientation
    norm = scatter.collections[0].norm
    
    bin_labels = ['[2,10)', '[10,20)', '[20,40)', '[40,80)', '[80,160)', '[160,240)', f"[240,{max_s})"]
    bin_centers = [6, 15, 30, 60, 120, 200, 320]

    if working_df[sizeCol].min() >= 10:
        bin_labels = ['[10,20)', '[20,40)', '[40,80)', '[80,160)', '[160,240)', f"[240,{max_s})"]
        bin_centers = [15, 30, 60, 120, 200, 320]

    pixel_sizes = min_s + (max_s - min_s) * norm(bin_centers)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label,
                markerfacecolor='gray', markersize=np.sqrt(ps))
        for label, ps in zip(bin_labels, pixel_sizes)
    ]

    if orientation == 'horizontal':
        # Invert legend elements order for top horizontal legend bar
        legend_elements = legend_elements[::-1]
        legend_kwargs = {
            'loc': 'lower center',
            'bbox_to_anchor': (0.5, 1.02),
            'ncols': len(legend_elements),
            'columnspacing': 1.5,
            'handletextpad': 0.5,
            'borderpad': 0.6,
        }
    else:
        legend_kwargs = {
            'loc': 'center left',
            'bbox_to_anchor': (1.02, 0.5),
            'ncols': 1,
            'labelspacing': 1.1,
            'handletextpad': 1.3,
            'borderpad': 0.5,
        }

    leg1 = ax.legend(
        handles=legend_elements,
        title=legend_title,
        frameon=True,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        **legend_kwargs,
    )
    leg1.get_frame().set_boxstyle("round", pad=0.5, rounding_size=0.2)

    # 5. Formatting labels & fontsizes based on orientation
    if orientation == 'vertical':
        x_label, x_fontsize = primaryLabel, primary_label_fontsize
        y_label, y_fontsize = secondaryLabel, secondary_label_fontsize
    else:
        x_label, x_fontsize = secondaryLabel, secondary_label_fontsize
        y_label, y_fontsize = primaryLabel, primary_label_fontsize

    if x_label:
        ax.set_xlabel(x_label, fontweight='bold', labelpad=15, fontsize=x_fontsize)
    else:
        ax.set_xlabel('')

    if y_label:
        ax.set_ylabel(y_label, fontweight='bold', labelpad=15, fontsize=y_fontsize)
    else:
        ax.set_ylabel('')

    if title:
        ax.set_title(title, fontweight='bold', pad=15, fontsize=title_fontsize)

    # 6. Axis limits & tick locator formatting
    num_categories = len(sorted_types)
    max_data_sec = working_df[secondaryCol].max()
    next_multiple_of_10 = int(np.ceil(max_data_sec / 10.0) * 10)
    bottom_padding = - (next_multiple_of_10 * 0.04)

    if orientation == 'vertical':
        plt.xticks(rotation=45, ha='right')
        ax.set_xlim(-0.5, num_categories - 0.5)
        ax.set_ylim(bottom=bottom_padding, top=next_multiple_of_10)
        ax.yaxis.set_major_locator(MultipleLocator(10))
    else:
        ax.set_ylim(-0.5, num_categories - 0.5)
        ax.set_xlim(left=bottom_padding, right=next_multiple_of_10)
        ax.xaxis.set_major_locator(MultipleLocator(10))

    ax.grid(True, linestyle='--', alpha=0.3)
    sns.despine(ax=ax)
    
    return ax