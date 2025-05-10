"""
BirdDMD - Dynamic Mode Decomposition for Bird Flight Analysis

This package provides tools for analyzing bird flight data using Dynamic Mode Decomposition (DMD).
It includes functionality for data loading, DMD computation, reconstruction, and analysis.

Main modules:
- data_handling: Core data loading and preprocessing
- dmd_core: Core DMD computation and analysis
- dmd_reconstruction: DMD reconstruction and mode modification
- pca_analysis: PCA-related analysis functions
- io: File input/output operations
- exceptions: Custom exception classes
- visualisation: Plotting and visualisation functions
"""

# Core DMD functionality
from .dmd_core import (
    run_dmd,
    run_sequence_dmd,
    run_marker_dmd,
    run_timeseries_dmd,
    run_single_wingbeat_dmd,
    dmd_loop_seqs,
    reorder_dmd_results
)

# DMD reconstruction and mode modification
from .dmd_reconstruction import (
    reconstruct_dmd,
    reconstruct_specific_modes,
    run_forecast,
    run_forecast_with_modified_modes,
    modify_mode_frequencies
)

# Data handling and preprocessing
from .data_handling import (
    load_bird_data,
    load_sequence_data,
    remove_time_duplicates,
    get_average_shape,
    normalise_data,
    add_average_shape,
    reshape_data,
    spline_interpolation
)

# PCA analysis
from .pca_space import (
    make_unilateral_keypoints,
    project_into_pca_space,
    project_into_coordinate_space,
    create_scores_info_df
)

# I/O operations
from .io import (
    save_sequence_results,
    load_sequence_results,
    load_dmd_results
)

# Visualisation
from .visualisation import (
    plot_markers_overDist,
    plot_2d_markers,
    plot_single_sequence,
    plot_amplitude_ranking,
    plot_score_multi_PCs
)

# Exceptions
from .exceptions import (
    BirdDMDError,
    ValidationError,
    ComputationError,
    DataError,
    ShapeError,
    IOError,
    FileNotFoundError,
    DataValidationError,
    ConfigurationError,
    InterpolationError,
    ReconstructionError,
    ModeError,
    EigenvalueError
)

# Version
__version__ = "0.1.0"

# Define public API
__all__ = [
    # Core DMD
    'run_dmd',
    'run_sequence_dmd',
    'run_marker_dmd',
    'run_timeseries_dmd',
    'run_single_wingbeat_dmd',
    'dmd_loop_seqs',
    
    # Reconstruction
    'reconstruct_dmd',
    'reconstruct_specific_modes',
    'run_forecast',
    'run_forecast_with_modified_modes',
    'modify_mode_frequencies',
    'reorder_dmd_results',
    
    # Data handling
    'load_bird_data',
    'load_sequence_data',
    'remove_time_duplicates',
    'get_average_shape',
    'normalise_data',
    'add_average_shape',
    'reshape_data',
    'spline_interpolation',
    
    # PCA analysis
    'make_unilateral_keypoints',
    'project_into_pca_space',
    'project_into_coordinate_space',
    'create_scores_info_df',
    
    # I/O operations
    'save_sequence_results',
    'load_sequence_results',
    'load_dmd_results',
    
    # Visualisation
    'plot_markers_overDist',
    'plot_2d_markers',
    'plot_single_sequence',
    'plot_amplitude_ranking',
    'plot_score_multi_PCs',
    
    # Exceptions
    'BirdDMDError',
    'ValidationError',
    'ComputationError',
    'DataError',
    'ShapeError',
    'IOError',
    'FileNotFoundError',
    'DataValidationError',
    'ConfigurationError',
    'InterpolationError',
    'ReconstructionError',
    'ModeError',
    'EigenvalueError'
]