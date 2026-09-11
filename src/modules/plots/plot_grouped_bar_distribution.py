import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, Optional, Iterable, List, Tuple, Any


def addArrowsToAlignedPositions(  ax           : plt.Axes
                                , df           : pd.DataFrame
                                , rna_col      : str
                                , target_ids   : Iterable[int]
                                , offset       : float = 0.08 ) -> None:
    """
    Helper function to dynamically place thick black arrows pointing downward above 
    specific bar groups based on matched 'label_seq_id_aligned' identifiers.

    Parameters:

                * ax :          a matplotlib Axes. The target axes where the bars are drawn.
                * df :          a pandas DataFrame. The dataset containing coordinates metadata.
                * rna_col :     a string. The column name serving as the X-axis labels.
                * target_ids :  an iterable of integers. The structural values to be highlighted.
                * offset :      a float. Vertical displacement for the arrow shaft text boundary.

    Returns: nothing
    """
    if not target_ids or not ax.containers:
        return

    # Extract structural text strings assigned to the dynamic graphical X-ticks
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Calculate mapping intersections between actual rows and positions within x_labels
    target_indices = set()
    for target_id in target_ids:
        matched_labels = df[df['label_seq_id_aligned'] == target_id][rna_col].unique()
        for label_val in matched_labels:
            if str(label_val) in x_labels:
                target_indices.add(x_labels.index(str(label_val)))

    # Process matplotlib bar elements to append annotations on mapped locations
    for i, bar in enumerate(ax.containers[0]):
        if i in target_indices:
            # Compute absolute graphical coordinates for the target rectangle block
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()
            
            # Prevent offset errors if height evaluation yields negative or zero bounds
            arrow_y = y_pos if (pd.notna(y_pos) and y_pos > 0) else 0.02
            
            # Annotate block elements with a thick black directional pointer downward
            ax.annotate(
                '', 
                xy=(x_pos, arrow_y), 
                xytext=(x_pos, arrow_y + offset),  
                arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.03, width=3, headwidth=9)
            )


def plotGroupedBarDistribution(df                     : pd.DataFrame
                            ,  x_col                  : str
                            ,  y_col                  : str
                            ,  hue_col                : str
                            ,  palette                : Any
                            ,  xlabel                 : Optional[str]               = None
                            ,  ylabel                 : Optional[str]               = None
                            ,  title                  : Optional[str]               = None
                            ,  legend_title           : Optional[str]               = None
                            ,  x_order                : Optional[List]              = None
                            ,  hue_order              : Optional[List]              = None
                            ,  use_log                : bool                        = False
                            ,  y_lim                  : Optional[Tuple[float, float]] = None
                            ,  y_step_min             : Optional[float]             = 0.2
                            ,  figsize                : Tuple[float, float]         = (20, 5)
                            ,  legend_panel_loc       : Optional[str]               = None
                            ,  title_fontsize         : Optional[float]             = None
                            ,  legend_title_fontsize  : Optional[float]             = None
                            ,  legend_labels_fontsize : Optional[float]             = None
                            ,  xlabel_fontsize        : Optional[float]             = None
                            ,  ylabel_fontsize        : Optional[float]             = None
                            ,  xticks_fontsize        : Optional[float]             = None
                            ,  yticks_fontsize        : Optional[float]             = None
                            ,  arrow_targets          : Optional[Iterable[int]]     = None
                            ,  ax                     : Optional[plt.Axes]          = None ) -> plt.Axes:
    """
    Generates a highly customizable, grouped categorical barplot using Seaborn.
    Supports dynamic axis scaling, custom step sizes, custom ordering, background grids,
    font size controls, legend placement, and arrow annotations via sequential alignment hooks.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Coerce frequency/numerical column to numeric values safely
    df = df.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

    # Render the categorical grouped barplot using Seaborn
    sns.barplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        palette=palette,
        order=x_order,
        hue_order=hue_order,
        ax=ax
    )

    # Configure title and axis labels with bold formatting
    if title:
        ax.set_title(title, fontweight='bold', fontsize=title_fontsize)
    else:
        ax.set_title('')

    if xlabel:
        ax.set_xlabel(xlabel, labelpad=20, fontweight='bold', fontsize=xlabel_fontsize)
    else:
        ax.set_xlabel('', labelpad=20)

    if ylabel:
        ax.set_ylabel(ylabel, labelpad=15, fontweight='bold', fontsize=ylabel_fontsize)
    else:
        ax.set_ylabel('', labelpad=15)

    # Configure tick params, rotation, and font sizes
    ax.tick_params(axis='x', rotation=45, labelsize=xticks_fontsize)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    ax.tick_params(axis='y', labelsize=yticks_fontsize)

    # Configure background grid
    ax.grid(axis='y')
    ax.set_axisbelow(True)

    # Customize legend formatting, title, and location
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_kwargs = {}
        if legend_panel_loc is not None:
            legend_kwargs['loc'] = legend_panel_loc

        leg = ax.legend(
            handles=handles,
            labels=labels,
            title=legend_title if legend_title is not None else '',
            fontsize=legend_labels_fontsize,
            title_fontsize=legend_title_fontsize,
            **legend_kwargs
        )
        if legend_title is None and leg:
            leg.set_title('')

    # Handle Y-axis scaling, boundaries, and step locator
    if use_log:
        ax.set_yscale('log')
    else:
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if y_step_min is not None and y_step_min > 0:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step_min))

    # Inject informational pointer cues using the helper function
    addArrowsToAlignedPositions(ax=ax, df=df, rna_col=x_col, target_ids=arrow_targets)

    plt.tight_layout()
    return ax