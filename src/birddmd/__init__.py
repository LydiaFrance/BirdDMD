"""
BirdDMD: Using DMD on bird wing data. 
"""
from __future__ import annotations

from .ProcessData import replace_rot_xyz_name, get_flight_modes, subset_by, get_column_names
from .DMDFigures import plot_markers_overDist, plot_2d_markers, plot_single_sequence
__all__ = ("__version__",)
__version__ = "0.1.0"
