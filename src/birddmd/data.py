"""Data validation, normalisation, binning, and reshaping.

This module provides dataset-agnostic utilities for preparing motion-capture
data before DMD analysis.  Hawk-specific file loading lives in `hawk`.

Functions
---------
validate_marker_data
    Check array shape and flatten to ``(n_frames, n_coords)``.
load_sequence_data
    Extract one sequence from a DataFrame by ``seqID``.
remove_time_duplicates
    Drop duplicate frames from a DataFrame.
normalise_data
    Centre data by subtracting a mean shape.
add_average_shape
    Inverse of ``normalise_data``.
bin_dataframe_means
    Temporal/spatial binning returning per-bin means.
spline_interpolation
    Cubic-spline onto evenly spaced time points.
expand_time_sequence
    Create an extended, evenly spaced time array.
expand_marker_sequence
    Repeat frames to fill an expanded time array.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from ._constants import DEFAULT_BIN_SIZE, N_SPATIAL, NDIM_2D, NDIM_3D

# ── Validation ──────────────────────────────────────────────────────


def validate_marker_data(data: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Validate marker data shape and flatten.

    Parameters
    ----------
    data : np.ndarray
        Either ``(n_frames, n_markers, 3)`` or ``(n_frames, n_markers*3)``.

    Returns
    -------
    data_flat : np.ndarray
        Shape ``(n_frames, n_coords)``.
    n_frames : int
    n_markers : int
    n_coords : int

    Raises
    ------
    ValueError
        If *data* is not 2-D or 3-D, or the last axis is not 3.
    """
    if data.ndim == NDIM_3D:
        n_frames, n_markers, n_spatial = data.shape
        if n_spatial != N_SPATIAL:
            msg = f"Expected {N_SPATIAL} coordinates per marker, got {n_spatial}"
            raise ValueError(msg)
        n_coords = n_markers * N_SPATIAL
        return data.reshape(n_frames, -1), n_frames, n_markers, n_coords

    if data.ndim == NDIM_2D:
        n_frames, n_coords = data.shape
        if n_coords % N_SPATIAL != 0:
            msg = f"Number of coordinates ({n_coords}) must be divisible by {N_SPATIAL}"
            raise ValueError(msg)
        return data, n_frames, n_coords // N_SPATIAL, n_coords

    msg = f"Data must be 2-D or 3-D, got {data.ndim}-D with shape {data.shape}"
    raise ValueError(msg)


# ── Sequence extraction ─────────────────────────────────────────────


def remove_time_duplicates(
    df: pd.DataFrame,
    column_name: str = "frameID",
) -> pd.DataFrame:
    """Drop duplicate rows based on *column_name*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (not modified in place).
    column_name : str
        Column used to detect duplicates.

    Returns
    -------
    pd.DataFrame
        De-duplicated copy, index reset.
    """
    out = df.drop_duplicates(subset=[column_name], keep="first")
    return out.reset_index(drop=True)


def load_sequence_data(
    df: pd.DataFrame,
    seqID: str,
    marker_column_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract marker coordinates and timestamps for one *seqID*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``'seqID'`` column, a ``'time'`` column, and the
        columns listed in *marker_column_names*.
    seqID : str
        Sequence identifier to filter on.
    marker_column_names : np.ndarray
        Column names for the marker coordinates.

    Returns
    -------
    markers : np.ndarray
        Shape ``(n_frames, n_coords)``, dtype float64.
    times : np.ndarray
        Shape ``(n_frames,)``, dtype float64.
    """
    rows = df["seqID"] == seqID
    markers = df.loc[rows, marker_column_names].to_numpy(dtype=np.float64)
    times = df.loc[rows, "time"].to_numpy(dtype=np.float64)
    return markers, times


# ── Normalisation ───────────────────────────────────────────────────


def normalise_data(markers: np.ndarray, average_shape: np.ndarray) -> np.ndarray:
    """Centre data by subtracting *average_shape*.

    Parameters
    ----------
    markers : np.ndarray
        Shape ``(n_frames, n_markers, 3)`` or ``(n_frames, n_coords)``.
    average_shape : np.ndarray
        Mean shape to subtract (broadcast-compatible).

    Returns
    -------
    np.ndarray
        Centred data, same shape as *markers*.
    """
    if markers.ndim == NDIM_3D:
        return markers - average_shape
    if markers.ndim == NDIM_2D:
        return markers - average_shape.reshape(1, -1)
    msg = f"Expected 2-D or 3-D marker data, got shape {markers.shape}"
    raise ValueError(msg)


def add_average_shape(data: np.ndarray, average_shape: np.ndarray) -> np.ndarray:
    """Add the mean shape back to centred data (inverse of ``normalise_data``)."""
    return data + average_shape


# ── Binning ─────────────────────────────────────────────────────────


def bin_dataframe_means(
    dataframe: pd.DataFrame,
    x_axis: str = "HorzDistance",
    bin_size: float = DEFAULT_BIN_SIZE,
    numeric_cast_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Bin a DataFrame along *x_axis* and return per-bin means.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data.
    x_axis : str
        Column used for binning.
    bin_size : float
        Width of each bin.
    numeric_cast_columns : list of str, optional
        Columns to cast to ``float`` before grouping.

    Returns
    -------
    pd.DataFrame
        One row per bin centre with mean values for numeric columns.
    """
    x_min = dataframe[x_axis].min()
    x_max = dataframe[x_axis].max()
    bins = np.arange(x_min, x_max + bin_size, bin_size)
    bins = np.around(bins, 3)
    bin_centres = bins[:-1] + bin_size / 2

    df = dataframe.copy()
    df["bins"] = pd.cut(df[x_axis], bins.tolist(), right=False, include_lowest=True)

    if numeric_cast_columns is not None:
        for col in numeric_cast_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)

    grouped = df.groupby("bins", observed=True).mean(numeric_only=True)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        grouped[col] = df.groupby("bins", observed=True)[col].first()

    grouped["time"] = bin_centres[: len(grouped)]
    grouped["seqID"] = "binned"
    return grouped


# ── Interpolation ───────────────────────────────────────────────────


def spline_interpolation(
    times: np.ndarray,
    markers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-spline interpolation onto evenly spaced time points.

    Parameters
    ----------
    times : np.ndarray
        Original (possibly unevenly spaced) time vector.
    markers : np.ndarray
        Shape ``(n_frames, n_coords)``.

    Returns
    -------
    new_times : np.ndarray
    new_markers : np.ndarray
    """
    new_times = np.linspace(times[0], times[-1], num=len(times))
    new_markers = np.zeros((len(new_times), markers.shape[1]))
    for i in range(markers.shape[1]):
        cs = CubicSpline(times, markers[:, i])
        new_markers[:, i] = cs(new_times)
    return new_times, new_markers


# ── Time / marker expansion ────────────────────────────────────────


def expand_time_sequence(
    times: np.ndarray,
    expansion_factor: float = 3.0,
) -> np.ndarray:
    """Create an expanded, evenly spaced time sequence.

    Parameters
    ----------
    times : np.ndarray
        Original time vector.
    expansion_factor : float
        Multiply the end time and frame count by this factor.

    Returns
    -------
    np.ndarray
        Evenly spaced times from ``times[0]`` to
        ``times[-1] * expansion_factor``.
    """
    return np.linspace(
        times[0],
        times[-1] * expansion_factor,
        len(times) * int(expansion_factor),
    )


def expand_marker_sequence(
    times: np.ndarray,
    markers: np.ndarray,
    expanded_times: np.ndarray,
) -> np.ndarray:
    """Repeat marker frames to fill *expanded_times*.

    Parameters
    ----------
    times : np.ndarray
        Original time vector.
    markers : np.ndarray
        Original marker data, shape ``(n_frames, ...)``.
    expanded_times : np.ndarray
        Target time vector (typically from ``expand_time_sequence``).

    Returns
    -------
    np.ndarray
        Expanded markers with the same trailing dimensions as *markers*.
    """
    expanded = []
    idx = 0
    for t in expanded_times:
        while idx < len(times) - 1 and t >= times[idx + 1]:
            idx += 1
        expanded.append(markers[min(idx, len(markers) - 1)].copy())
    return np.array(expanded)
