import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict


def plotRnaConformerLigandBidirectional(
    df              : pd.DataFrame,
    rna_col         : str,
    ligand_col      : str,
    rna_palette     : Dict[str, str],
    xlabel          : str,
    ylabel          : str,
    title           : str                = None,
    rna_order       : Optional[list]     = None,
    alpha_without   : float              = 0.4,
    alpha_with      : float              = 1.0,
    ax              : Optional[plt.Axes] = None,
    figsize         : tuple              = (6, 6)
) -> plt.Axes:
    """
    Plots a 100% bidirectional vertical bar chart optimized for a 6x6 square layout.
    Bar width and text size are tightly calibrated to avoid empty space and overlaps.
    """
    # 1. Configurar contexto académico y forzar tamaño cuadrado si no se provee un eje
    sns.set_theme(context='paper', style='ticks')
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) # Forzado a 6x6 pulgadas
    
    # 2. Limpiar nulos
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=[rna_col, ligand_col])
    
    # 3. Calcular porcentajes por tipo de ARN
    ct = pd.crosstab(plot_df[rna_col], plot_df[ligand_col])
    if rna_order:
        ct = ct.reindex(rna_order, fill_value=0)
    else:
        ct = ct.sort_index()
        
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    for col in [False, True]:
        if col not in ct_pct.columns:
            ct_pct[col] = 0.0

    categories = ct_pct.index
    x_pos = np.arange(len(categories))
    
    # CALIBRACIÓN PARA 6x6: Un ancho de 0.65 a 0.7 evita espacios vacíos excesivos en lienzos angostos
    bar_width = 0.68

    vals_without = ct_pct[False].values
    vals_with = ct_pct[True].values

    # 4. Dibujar barras bidireccionales verticales
    for i, cat in enumerate(categories):
        color = rna_palette.get(cat, '#gray')
        
        # Abajo (Without ligand)
        ax.bar(x_pos[i], -vals_without[i], width=bar_width, 
               color=color, alpha=alpha_without, edgecolor='black', linewidth=0.8)
        
        # Arriba (With ligand)
        ax.bar(x_pos[i], vals_with[i], width=bar_width, 
               color=color, alpha=alpha_with, edgecolor='black', linewidth=0.8)

    # Línea divisoria central en el cero horizontal
    ax.axhline(0, color='black', linewidth=1.0, zorder=3)

    # 5. Cálculo dinámico de límites adaptado a la compresión del lienzo cuadrado
    max_barra_superior = max(vals_with) + 8 if len(vals_with) > 0 else 100
    max_barra_inferior = max(vals_without) + 8 if len(vals_without) > 0 else 100
    
    padding_titulo = 16
    y_pos_titulo_superior = max(max_barra_superior, 100) + padding_titulo
    y_pos_titulo_inferior = -(max(max_barra_inferior, 100) + padding_titulo)
    
    ax.set_ylim(y_pos_titulo_inferior - 8, y_pos_titulo_superior + 8)  

    # Formatear el eje Y
    ticks = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
    tick_labels = [f'{abs(t)}%' for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel(ylabel, labelpad=15, fontsize=9, fontweight='bold')
    
    # 6. Etiquetas numéricas EXTERNAS ligeramente reducidas (fontsize=7.5) para que entren en el ancho de la barra
    # Bloque superior: With ligand
    for i, pct in enumerate(vals_with):
        if pct > 0:
            label_text = f'{pct:.1f}%'
            y_text = pct + 2.0  
            ax.text(x_pos[i], y_text, label_text, ha='center', va='bottom', 
                    fontsize=7.5, fontweight='bold', color='black')

    # Bloque inferior: Without ligand
    for i, pct in enumerate(vals_without):
        if pct > 0:
            label_text = f'{pct:.1f}%'
            y_text = -pct - 2.0  
            ax.text(x_pos[i], y_text, label_text, ha='center', va='top', 
                    fontsize=7.5, fontweight='bold', color='black')

    # 7. Formatear eje X optimizado para evitar cortes en figuras cuadradas
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9.0)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=8)

    if title:
        ax.set_title(title, pad=35, fontweight='bold', fontsize=11)

    # 8. Textos de sección universales
    centro_x = (len(categories) - 1) / 2.0
    ax.text(centro_x, y_pos_titulo_superior, 'with ligand ▲', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#4f4f4f')
    ax.text(centro_x, y_pos_titulo_inferior, 'without ligand ▼', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#4f4f4f')

    # Rejillas y despine
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.grid(False, axis='x')
    sns.despine(ax=ax, bottom=False, left=False)

    # Ajustar márgenes para que las etiquetas diagonales inferiores de 6x6 no se corten al guardar
    plt.tight_layout()

    return ax
