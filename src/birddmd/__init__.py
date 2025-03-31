"""
birddmd: Using DMD on bird wing data. 
"""
from __future__ import annotations

from .ProcessData import replace_rot_xyz_name, get_flight_modes, subset_by, get_column_names
from .DMDFigures import plot_markers_overDist, plot_2d_markers, plot_single_sequence, plot_score_multi_PCs
from .runDMD import (
    run_single_wingbeat_dmd, dmd_loop_seqs, run_DMD_sequence, 
    load_bird_data, remove_time_duplicates, load_sequence_data, 
    get_average_shape, normalise_data, add_average_shape, reshape_data,
    perform_dmd, run_forecast, reorder_dmd_results, reconstruct_dmd,
    reconstruct_specific_modes, make_unilateral_keypoints, project_into_pca_space,
    create_scores_info_df, 
    save_sequence_results, load_sequence_results, load_dmd_results)

from .aggregateDMD import get_every_sequence_result, plot_aggregate_DMD_histogram

__all__ = ("__version__", "replace_rot_xyz_name", "get_flight_modes", 
           "subset_by", "get_column_names", "plot_markers_overDist", 
           "plot_2d_markers", "plot_single_sequence", "plot_score_multi_PCs", 
           "run_single_wingbeat_dmd", "dmd_loop_seqs", "run_DMD_sequence",
           "load_bird_data", "remove_time_duplicates", "load_sequence_data", 
           "get_average_shape", "normalise_data", "add_average_shape", 
           "reshape_data", "perform_dmd", "run_forecast", "reorder_dmd_results", 
           "reconstruct_dmd", "reconstruct_specific_modes", "make_unilateral_keypoints",
           "project_into_pca_space", "create_scores_info_df", "save_sequence_results", 
           "load_sequence_results", "load_dmd_results", "get_every_sequence_result", 
           "plot_aggregate_DMD_histogram")
__version__ = "0.1.0"