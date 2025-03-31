"""
birddmd: Using DMD on bird wing data. 
"""
from __future__ import annotations

from .ProcessData import replace_rot_xyz_name, get_flight_modes, subset_by, get_column_names
from .DMDFigures import plot_markers_overDist, plot_2d_markers, plot_single_sequence, plot_score_multi_PCs
from .runDMD import (
    run_single_wingbeat_dmd,
    load_bird_data, remove_time_duplicates, 
    load_sequence_data, normalise_data, perform_dmd, 
    run_forecast, reorder_dmd_results, reconstruct_dmd,
    save_sequence_results, load_sequence_results, load_dmd_results)

from .aggregateDMD import get_every_sequence_result, plot_aggregate_DMD_histogram

__all__ = ("__version__", "replace_rot_xyz_name", "get_flight_modes", 
           "subset_by", "get_column_names", "plot_markers_overDist", 
           "plot_2d_markers", "plot_single_sequence", "plot_score_multi_PCs", "run_single_wingbeat_dmd",
             "load_bird_data", 
           "remove_time_duplicates", "load_sequence_data", "normalise_data", 
           "perform_dmd", "run_forecast", "reorder_dmd_results", "reconstruct_dmd",
            "save_sequence_results", "load_sequence_results", "load_dmd_results", 
            "get_every_sequence_result", "plot_aggregate_DMD_histogram")
__version__ = "0.1.0"