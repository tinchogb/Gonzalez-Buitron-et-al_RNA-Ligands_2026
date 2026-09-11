import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict

def plotRnaPartnerFrequencySum(df            : pd.DataFrame
                            ,  values        : str
                            ,  rna_col       : str
                            ,  partner_col   : str
                            ,  palette       : Dict[str, str]
                            ,  xlabel        : str
                            ,  ylabel        : str
                            ,  title         : str                = None
                            ,  legend_title  : str                = None
                            ,  rna_order     : Optional[list]     = None
                            ,  partner_order : Optional[list]     = None
                            ,  use_log       : bool               = True
                            ,  ax            : Optional[plt.Axes] = None
                            ,  figsize       : tuple            = (10, 5.5) ) -> plt.Axes:
    """
    Plot a grouped barplot indicating the total sum of frequency per partner type,
    subdivided by RNA type categories, supporting both linear and logarithmic Y-axes
    by calculating text label coordinates independently from the graphical bar elements.

    Parameters:

                * df :            a pandas DataFrame. The dataset containing RNA and partner metadata.
                * values :        a string. The column containing frequency values.
                * rna_col :       a string. The column name for the RNA types (e.g., 'rna_type_cdrna_grouped').
                * partner_col :   a string. The column name for the partner types (e.g., 'partner_rcsb_type_grouped').
                * palette :       a dictionary. Corresponds to the color palette mapped to partner types.
                * xlabel :        a string. Corresponds to the label for the x-axis.
                * ylabel :        a string. Corresponds to the label for the y-axis.
                * title :         a string. Corresponds to the title of the plot. Default is None.
                * legend_title :  a string. Corresponds to the title of the legend. Default is None.
                * rna_order :     a list. The order of RNA types on the x-axis. Default is None.
                * partner_order : a list. The order of partner bars inside each RNA group. Default is None.
                * use_log :       a boolean. If True, the Y-axis will be logarithmic; if False, linear. Default is True.
                * ax :            a matplotlib Axes. Existing axes to draw onto. Default is None.
                * figsize :       a tuple. The size of the figure in inches (width, height). Default is (10, 5.5).

    Observations:

                * Explicitly coerces the frequency column to numeric to handle potential TypeErrors.
                * Aggregates frequency data using a sum grouped by both RNA and partner categories.
                * Guarantees that all combinations of RNA and partner types exist in the dataset, filling missing combinations with zero frequency.
                * Supports both linear and logarithmic Y-axes, dynamically adjusting label positions for readability.
                * Dynamically calculates and injects bar labels based on the current scale environment.

    Returns:

                * plt.Axes : The matplotlib Axes object containing the finalized grouped barplot.

    """
    # 1. Set academic context
    sns.set_theme(context='paper', style='ticks')
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
    
    # 2. Make a copy to prevent SettingWithCopyWarning and coerce type safely
    plot_df = df.copy()
    plot_df[values] = pd.to_numeric(plot_df[values], errors='coerce')
    
    # 3. Group and aggregate by BOTH columns to calculate cross-frequencies
    df_grouped = (
        plot_df.dropna(subset=[values])
        .groupby([rna_col, partner_col])[values]
        .sum()
        .reset_index(name='total_frequency')
    )
    
    # Guarantee all combinations of RNA and Partner exist with their real 0 values
    full_idx = pd.MultiIndex.from_product(
        [rna_order if rna_order else df_grouped[rna_col].unique(),
        partner_order if partner_order else df_grouped[partner_col].unique()],
        names=[rna_col, partner_col]
    )
    df_grouped = df_grouped.set_index([rna_col, partner_col]).reindex(full_idx, fill_value=0).reset_index()

    # 4. Render grouped bars using PURE, unaltered data
    sns.barplot(
        data=df_grouped,
        x=rna_col,
        y='total_frequency',  # Real data! No forced values
        hue=partner_col,
        order=rna_order,
        hue_order=partner_order,
        palette=palette,
        alpha=1.0,
        ax=ax
    )
    
    # 5. Apply scale and bounding configurations dynamically
    if use_log:
        ax.set_yscale('log')
        # Standardize limits without pushing the baseline away from standard log ticks
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(bottom=0.11, top=current_ymax * 3)  # Push top limit slightly for label visibility
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.2)
    else:
        ax.set_yscale('linear')
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(bottom=0, top=current_ymax * 1.15) 
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.2)

    ax.set_autoscale_on(False)
    
    # 6. Calculate and inject the bar labels dynamically
    # We fetch the bottom boundary of the current scale environment for text placement 
    y_floor = ax.get_ylim()[0] 

    for container in ax.containers:
        for rect in container:
            # Extract the native geometric data of each bar slice
            x_pos = rect.get_x() + rect.get_width() / 2.0
            y_val = rect.get_height()
            
            # Matplotlib inputs NaN heights for invalid log(0) components
            if np.isnan(y_val) or y_val == 0:
                label_text = "0"
                # Place text at the absolute visual bottom line of the current scale
                y_pos = y_floor 
            else:
                label_text = f"{y_val:.0f}"
                y_pos = y_val

            # Draw the annotation directly into the system coordinate layout
            ax.annotate(
                label_text,
                xy=(x_pos, y_pos),
                xytext=(0, 4),  # 4 points padding upward
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90,
                color='black',
                fontweight='normal'
            )
    
    # 7. Format axes labels, titles, and legend
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    else:
        ax.set_xlabel('', labelpad=0)
        
    ax.set_ylabel(ylabel, labelpad=15, fontsize=10, fontweight='bold')
    if title:
        ax.set_title(title, pad=15, fontweight='bold')

    ax.legend(title     = legend_title
            , loc       = 'upper right'
            , frameon   = True
            , facecolor = 'white'
            , edgecolor = 'black' )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.grid(False, axis='x')  
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    return ax