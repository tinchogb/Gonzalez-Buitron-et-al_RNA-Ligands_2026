from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt

from .save_plot import savePlot


def savePlotDecorator(dpi           : int               = 300
                    , format        : Optional[str]     = None
                    , transparent   : bool              = False
                    , tight         : bool              = True
                    , pad_inches    : float             = 0.1
                    , output_dir    : Union[str, Path]  = "figures"
                    , filename_fmt  : str               = "{func_name}{suffix}"
                    , overwrite     : bool              = True
                    , verbose       : bool              = True ) -> Callable:
    """
    Decorator that wraps any plotting function returning a plt.Axes, a sequence/array
    of plt.Axes, a dictionary of plt.Axes (e.g., plotClusterUpSet), or complex
    composite return values, and optionally saves the figure after plotting.
    The wrapped function gains two extra keyword arguments, save_path and
    save_dpi, without altering its original signature.

    Parameters:

                * dpi :          an integer. Default resolution in dots per inch
                                 when the caller does not override it via
                                 save_dpi. Default is 300.
                * format :       a string or None. Default output format used
                                 when the caller does not provide a save_path
                                 with a recognizable extension.
                                 Default is None.
                * transparent :  a boolean. Default background transparency.
                                 Default is False.
                * tight :        a boolean. Whether to apply tight cropping on
                                 save. Default is True.
                * pad_inches :   a float. Padding for tight cropping.
                                 Default is 0.1.
                * output_dir :   a string or Path. Directory where figures are
                                 saved when the caller does not provide an
                                 explicit save_path. Created automatically if
                                 missing. Default is "figures".
                * filename_fmt : a string. Template used to auto-generate the
                                 filename when the caller does not provide
                                 save_path. Available placeholders are
                                 {func_name} (the plotting function name) and
                                 {suffix} (an optional caller-provided tag).
                                 Default is "{func_name}{suffix}".
                * overwrite :    a boolean. Default overwrite behaviour passed
                                 to savePlot. Default is True.
                * verbose :      a boolean. Whether to print the saved path.
                                 Default is True.

    Observations:

                * The wrapped plotting function is called first; saving only
                  happens if the caller passes save_path (explicit path) or
                  save=True (auto-generated path), so interactive use in
                  Jupyter is unaffected.
                * Recursively resolves target Axes objects from single Axes instances,
                  arrays, sequences, dictionaries, or composite tuples (e.g., return_results=True).
                * The returned Axes or Axes container is always passed through
                  unchanged, so the caller can still modify or re-save the figure afterwards.
                * Default export parameters are centralized here; per-call
                  overrides are possible via save_path/save_dpi and through
                  the decorator arguments applied at decoration time.
                * functools.wraps preserves the original function name,
                                  signature, and docstring for introspection.

    Returns:

                * Callable : The wrapped plotting function.

    """
    def _extract_axes(obj: Any) -> Optional[plt.Axes]:
        """Recursively search for an object containing a .figure attribute."""
        if hasattr(obj, "figure"):
            return obj
        if isinstance(obj, dict):
            for item in obj.values():
                found = _extract_axes(item)
                if found is not None:
                    return found
        elif isinstance(obj, (tuple, list, set)):
            for item in obj:
                found = _extract_axes(item)
                if found is not None:
                    return found
        elif hasattr(obj, "flat"):
            for item in obj.flat:
                found = _extract_axes(item)
                if found is not None:
                    return found
        return None

    def decorator(plot_func: Callable) -> Callable:

        @wraps(plot_func)
        def wrapper(*args, **kwargs) -> Union[plt.Axes, Tuple[plt.Axes, ...], Dict[str, plt.Axes], Any]:

            # Extract the save-related arguments injected by this decorator
            save_path   = kwargs.pop("save_path", None)
            save_dpi    = kwargs.pop("save_dpi", dpi)
            save        = kwargs.pop("save", False)
            save_suffix = kwargs.pop("save_suffix", "")

            res = plot_func(*args, **kwargs)

            if save_path is not None or save:
                if save_path is None:
                    filename = filename_fmt.format(
                        func_name=plot_func.__name__,
                        suffix=save_suffix
                    )
                    ext = f".{format.lstrip('.')}" if format else ".png"
                    save_path = Path(output_dir) / f"{filename}{ext}"

                # Extract a valid Axes object to access the parent Figure
                target_ax = _extract_axes(res)

                if target_ax is None:
                    raise AttributeError(
                        f"Could not resolve a valid matplotlib Axes from the return value of {plot_func.__name__}."
                    )

                savePlot( ax          = target_ax
                        , filepath    = save_path
                        , dpi         = save_dpi
                        , format      = format
                        , transparent = transparent
                        , tight       = tight
                        , pad_inches  = pad_inches
                        , overwrite   = overwrite
                        , verbose     = verbose
                )

            return res

        return wrapper

    return decorator