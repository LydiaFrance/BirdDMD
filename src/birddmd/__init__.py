"""BirdDMD — Dynamic Mode Decomposition for bird flight analysis.

A minimal, scientist-friendly toolkit for decomposing motion-capture
time series into spatially coherent oscillatory modes.

Modules
-------
core
    ``DMDResult``, ``run_dmd``, ``detect_conjugate_pairs``,
    ``convergence_analysis``.
reconstruction
    ``reconstruct``, ``forecast``.
data
    Validation, normalisation, binning, reshaping.
stats
    ``compute_rmse``, ``filter_sequences``, ``variance_explained``.
plotting
    Dataset-agnostic DMD plots.
hawk
    Hawk-specific loading, wrappers, and plots.
"""

__version__ = "1.0.0"

# ── Core ────────────────────────────────────────────────────────────

from ._constants import (
    DEFAULT_MARKER_NAMES,
    HAWK_META_COLUMNS,
    MARKER_COLOURS,
    MARKER_POSITIONS,
)
from .core import (
    DMDResult,
    convergence_analysis,
    detect_conjugate_pairs,
    run_dmd,
)

# ── Data ────────────────────────────────────────────────────────────
from .data import (
    add_average_shape,
    bin_dataframe_means,
    expand_marker_sequence,
    expand_time_sequence,
    load_sequence_data,
    normalise_data,
    remove_time_duplicates,
    spline_interpolation,
)

# ── Hawk convenience layer ──────────────────────────────────────────
from .hawk import (
    MEAN_SHAPE_PATH,
    SAMPLES_DIR,
    batch_rmse_analysis,
    bin_hawk_dataframe,
    get_average_shape,
    load_bird_data,
    normalise_hawk_data,
    plot_hawk_2d,
    plot_hawk_markers,
    plot_hawk_markers_shaded,
    plot_single_sequence,
    run_hawk_dmd,
    run_sequence_dmd,
)

# ── Plotting ────────────────────────────────────────────────────────
from .plotting import (
    plot_amplitude_ranking,
    plot_convergence,
    plot_mode_dynamics,
)

# ── Reconstruction ──────────────────────────────────────────────────
from .reconstruction import forecast, reconstruct

# ── Stats ───────────────────────────────────────────────────────────
from .stats import compute_rmse, filter_sequences, variance_explained

# ── Public API ──────────────────────────────────────────────────────

__all__ = [
    "DEFAULT_MARKER_NAMES",
    "HAWK_META_COLUMNS",
    "MARKER_COLOURS",
    "MARKER_POSITIONS",
    "MEAN_SHAPE_PATH",
    "SAMPLES_DIR",
    "DMDResult",
    "add_average_shape",
    "batch_rmse_analysis",
    "bin_dataframe_means",
    "bin_hawk_dataframe",
    "compute_rmse",
    "convergence_analysis",
    "detect_conjugate_pairs",
    "expand_marker_sequence",
    "expand_time_sequence",
    "filter_sequences",
    "forecast",
    "get_average_shape",
    "load_bird_data",
    "load_sequence_data",
    "normalise_data",
    "normalise_hawk_data",
    "plot_amplitude_ranking",
    "plot_convergence",
    "plot_hawk_2d",
    "plot_hawk_markers",
    "plot_hawk_markers_shaded",
    "plot_mode_dynamics",
    "plot_single_sequence",
    "reconstruct",
    "remove_time_duplicates",
    "run_dmd",
    "run_hawk_dmd",
    "run_sequence_dmd",
    "spline_interpolation",
    "variance_explained",
]
