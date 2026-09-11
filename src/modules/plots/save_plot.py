from pathlib import Path
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt


def savePlot( ax               : plt.Axes
            , filepath         : Union[str, Path]
            , dpi              : int                      = 300
            , format           : Optional[str]            = None
            , transparent      : bool                     = False
            , tight            : bool                     = True
            , pad_inches       : float                    = 0.1
            , facecolor        : Optional[str]            = None
            , include_legend   : bool                     = True
            , metadata         : Optional[Dict[str, str]] = None
            , overwrite        : bool                     = True
            , close            : bool                     = False
            , verbose          : bool                     = True ) -> Path:
    """
    Save a matplotlib figure to disk with explicit control over resolution
    and export quality. Designed as the single delegation point for all
    plotting functions that return a plt.Axes object.

    Parameters:

                * ax :            a matplotlib Axes. The Axes returned by any
                                  plotting function (e.g., plotContactPositionProportion).
                                  The parent figure is resolved automatically via
                                  ax.figure.
                * filepath :      a string or Path. Destination path. The file
                                  extension (if *format* is None) determines the
                                  output format (e.g., ".png", ".pdf", ".svg",
                                  ".tiff").
                * dpi :           an integer. Resolution in dots per inch for the
                                  saved file. Use 150 for web/slides, 300 for
                                  reports and posters, 600 for print publications.
                                  Does not affect on-screen rendering in Jupyter.
                                  Default is 300.
                * format :        a string or None. Explicit output format
                                  ("png", "pdf", "svg", "tiff", ...). If None,
                                  the format is inferred from the file extension.
                                  Default is None.
                * transparent :   a boolean. Whether to save with a transparent
                                  background. Useful for overlaying figures on
                                  slides with non-white backgrounds.
                                  Default is False.
                * tight :         a boolean. Whether to crop the figure to the
                                  bounding box of all visible artists
                                  (bbox_inches="tight"). Strongly recommended for
                                  boxplots with significance brackets, which often
                                  extend beyond the nominal figure area.
                                  Default is True.
                * pad_inches :    a float. Extra padding around the figure when
                                  *tight* is True. Default is 0.1.
                * facecolor :     a string or None. Explicit figure background
                                  color. If None, the current figure facecolor is
                                  kept. Ignored when *transparent* is True.
                                  Default is None.
                * include_legend : a boolean. Whether to explicitly include the
                                  legend in the tight bounding box computation.
                                  Only relevant when *tight* is True and a legend
                                  was added manually outside the axes.
                                  Default is True.
                * metadata :      a dictionary or None. Key-value pairs embedded
                                  into the file metadata (supported for PNG and
                                  PDF, e.g., {"Title": ..., "Author": ...}).
                                  Default is None.
                * overwrite :     a boolean. If False, raises FileExistsError when
                                  the destination file already exists, preventing
                                  accidental loss of previously exported figures.
                                  Default is True.
                * close :         a boolean. Whether to close the figure after
                                  saving to free memory when producing many
                                  figures in a loop. Default is False.
                * verbose :       a boolean. Whether to print the path and
                                  effective pixel dimensions after saving.
                                  Default is True.

    Observations:

                * Extracts the parent figure from *ax* via ax.figure, so any
                  plotting function returning an Axes is compatible without
                  modification.
                * The final pixel size is determined by figsize x dpi; with
                  tight cropping enabled, the physical dimensions may shrink
                  slightly to fit all artists.
                * Vector formats (PDF, SVG) ignore *dpi* for line art and text,
                  which are stored as vectors; *dpi* only affects embedded
                  raster elements if any.
                * The file extension is normalized to lowercase, and the parent
                  directory is created automatically if it does not exist.

    Returns:

                * pathlib.Path : The resolved path of the saved file.

    """
    path = Path(filepath).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists and overwrite=False: {path}"
        )

    fig = ax.figure

    save_kwargs = dict(
        dpi=dpi,
        transparent=transparent,
        facecolor=facecolor,
        metadata=metadata,
    )

    if tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = pad_inches
        if include_legend:
            legend = ax.get_legend()
            if legend is not None:
                save_kwargs["bbox_extra_artists"] = (legend,)

    if format is not None:
        save_kwargs["format"] = format.lower()

    fig.savefig(path, **save_kwargs)

    if verbose:
        width_px, height_px = fig.canvas.get_width_height()
        print(f"Figure saved -> {path} ({width_px}x{height_px} px @ {dpi} DPI)")

    if close:
        plt.close(fig)

    return path