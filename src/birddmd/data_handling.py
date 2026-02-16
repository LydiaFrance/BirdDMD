"""
Core data loading and preprocessing functions for BirdDMD.

This module handles loading, validating, and preprocessing bird marker data.
It provides functions for data loading, shape validation, normalisation,
and basic data transformations.
"""

import os

import numpy as np
import pandas as pd
from morphing_birds import Hawk3D
from scipy.interpolate import CubicSpline

# Constants
N_SPATIAL_DIMS = 3
N_MARKERS_FULL = 8
N_MARKERS_HALF = 4
SAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples")
)


# Error handling
class DataError(Exception):
    """Base class for data-related errors."""


class ShapeError(DataError):
    """Raised when array shapes are invalid."""


class ValidationError(DataError):
    """Raised when data validation fails."""


def validate_marker_data(data: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Validate marker data shape and return dimensions.

    Args:
        data: Input data array

    Returns:
        tuple of (validated_data, n_frames, n_markers, total_coords)

    Raises:
        ShapeError: If data shape is invalid
    """
    if len(data.shape) not in [2, 3]:
        msg = (
            "Data must be 2D (n_frames, n_markers*3) or 3D (n_frames, n_markers, 3),"
            f"got shape {data.shape}"
        )
        raise ShapeError(msg)

    if len(data.shape) == N_SPATIAL_DIMS:
        n_frames, n_markers, n_coords = data.shape
        if n_coords != N_SPATIAL_DIMS:
            msg = f"Expected {N_SPATIAL_DIMS} coordinates per marker, got {n_coords}"
            raise ShapeError(msg)
        data_flat = data.reshape(n_frames, -1)
        total_coords = n_markers * n_coords
    else:
        n_frames, total_coords = data.shape
        if total_coords % N_SPATIAL_DIMS != 0:
            msg = (
                f"Number of coordinates ({total_coords}) must be divisible by"
                f"{N_SPATIAL_DIMS}"
            )
            raise ShapeError(msg)
        n_markers = total_coords // N_SPATIAL_DIMS
        data_flat = data

    return data_flat, n_frames, n_markers, total_coords


def validate_times(times: np.ndarray, n_frames: int) -> np.ndarray:
    """Validate time vector length matches data.

    Args:
        times: Time vector
        n_frames: Expected number of frames

    Returns:
        Validated time vector

    Raises:
        ValidationError: If time vector length doesn't match data
    """
    if times is None:
        return np.arange(n_frames)

    if len(times) != n_frames:
        msg = (
            f"Time vector length ({len(times)}) "
            f"must match number of frames ({n_frames})"
        )
        raise ValidationError(msg)

    return times


def validate_average_shape(
    avg_shape: np.ndarray, n_coords: int, is_timeseries: bool = False
) -> np.ndarray:
    """Validate average shape dimensions.

    Args:
        avg_shape: Average shape array
        n_coords: Expected number of coordinates (total coordinates for marker data)
        is_timeseries: Whether this is time series data (not marker data)

    Returns:
        Validated average shape

    Raises:
        ValidationError: If average shape dimensions are invalid
    """
    if avg_shape is None:
        if is_timeseries:
            return np.zeros((1, n_coords))
        msg = "Average shape must be provided for marker data"
        raise ValidationError(msg)

    avg_shape = avg_shape.reshape(1, -1)
    if not is_timeseries and avg_shape.shape[1] != n_coords:
        msg = (
            f"Average shape coordinates ({avg_shape.shape[1]})"
            f"must match data coordinates ({n_coords})"
        )
        raise ValidationError(msg)

    return avg_shape


def load_bird_data(
    bird_name: str,
    behaviour: str,
    perch_distance: str | None = None,
    bilateral: str = "Bilateral",
    turn: str = "Straight",
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load bird marker and info data into a DataFrame based on specified parameters.

    Args:
        bird_name: Name of the bird (e.g., 'Toothless')
        behaviour: Flight behaviour (e.g., 'Initial')
        perch_distance: Perch distance (e.g., '9m'). If None, assumes filename
                        doesn't specify distance
        bilateral: 'Bilateral' or 'Unilateral'
        turn: Turn direction ('Straight', 'Left', 'Right')
        verbose: Whether to print verbose output about dataframe shape and number
                 of sequences
    Returns:
        Tuple of (wingbeat_df, marker_column_names)

    Raises:
        FileNotFoundError: If data file not found
        ValidationError: If data format is invalid
    """
    # Construct the bilateral status string
    bilateral_status = "Bilateral" if bilateral else "Unilateral"

    if perch_distance is None:
        perch_distance = ""
        perch_distance_text = "all"
    else:
        perch_distance_text = perch_distance

    # Construct the file path
    if perch_distance == "9m":
        file_path = (
            f"{SAMPLES_DIR}/{behaviour}_{perch_distance}"
            f"{turn}Turn{bird_name}_{bilateral_status}.npz"
        )
        turn_text = f", with {turn.lower()} turns only"
    else:
        file_path = (
            f"{SAMPLES_DIR}/{behaviour}_{perch_distance}"
            f"{bird_name}_{bilateral_status}.npz"
        )
        turn_text = ""

    try:
        # Load the data from the specified file
        file = np.load(file_path, allow_pickle=True)
        marker_data = file["marker_data"]
        info_data = file["info_data"]

        # Load column names
        column_names = np.load(SAMPLES_DIR + "/ColumnNames.npz")
        marker_column_names = column_names["marker_column_names"]
        info_column_names = column_names["info_column_names"]

        # Create DataFrames for marker and info data
        marker_df = pd.DataFrame(marker_data, columns=marker_column_names)
        info_df = pd.DataFrame(info_data, columns=info_column_names)

        # Concatenate the DataFrames horizontally
        wingbeat_df = pd.concat([info_df, marker_df], axis=1)

        # Calculate the number of unique sequences
        n_sequences = wingbeat_df["seqID"].nunique()

        # Report the loaded DataFrame size
        if verbose:
            print(
                f"Loading {bird_name} {bilateral_status.lower()} data"
                f" with {perch_distance_text} perch distances{turn_text},"
                f"{behaviour.lower()}."
            )
            print(
                f"Dataframe with shape {wingbeat_df.shape} loaded."
                f"Number of sequences: {n_sequences}"
            )

        return wingbeat_df, marker_column_names

    except FileNotFoundError as e:
        msg = f"Data file not found: {file_path}"
        raise FileNotFoundError(msg) from e
    except Exception as e:
        msg = f"Error loading data: {e!s}"
        raise ValidationError(msg) from e


def remove_time_duplicates(
    df: pd.DataFrame, column_name: str = "frameID", verbose: bool = False
) -> pd.DataFrame:
    """
    Removes duplicate rows based on a column (typically 'frameID').

    Args:
        df: Input DataFrame
        column_name: Column to check for duplicates
        verbose: Whether to print the percentage of duplicates found

    Returns:
        DataFrame with duplicates removed and index reset
    """
    # Identify duplicates
    duplicates = df[df.duplicated(subset=[column_name], keep=False)]
    ratio_duplicates = (duplicates.shape[0] / df.shape[0]) * 100

    # Report percentage of duplicates
    if verbose and ratio_duplicates > 0:
        print(
            f"Duplicated frames are {np.round(ratio_duplicates, 2)}% of the total data."
        )

    # Remove duplicates
    df = df.drop_duplicates(subset=[column_name], keep="first")

    # Reset the index
    return df.reset_index(drop=True)


def load_sequence_data(
    df: pd.DataFrame, seqID: str, marker_column_names: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts marker coordinates and timestamps for a specific seqID from the
    main DataFrame.

    Args:
        df: DataFrame containing all sequences
        seqID: The specific sequence ID to extract
        marker_column_names: List of marker coordinate column names

    Returns:
        Tuple of (markers, times) where markers is shape (n_frames, n_coords)
        and times is shape (n_frames,)
    """
    condition = df["seqID"] == seqID
    markers = df[condition][marker_column_names].to_numpy().astype(np.float64)
    times = df[condition]["time"].to_numpy().astype(np.float64)
    return markers, times


def get_average_shape(nMarkers: int, mean_shape_path: str | None = None) -> np.ndarray:
    """
    Loads the mean hawk shape from a standard file based on the number of markers.

    Args:
        nMarkers: The number of markers (4 or 8)
        mean_shape_path: Path to the mean hawk shape CSV file.
                         If None, uses default path

    Returns:
        The mean shape as a (1, n_coords) NumPy array

    Raises:
        ValidationError: If number of markers is invalid
    """
    if mean_shape_path is None:
        mean_shape_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mean_hawk_shape.csv"
        )

    hawk3d = Hawk3D(mean_shape_path)
    if nMarkers == N_MARKERS_FULL:
        return hawk3d.markers
    if nMarkers == N_MARKERS_HALF:
        return hawk3d.right_markers

    msg = (
        f"Expected {N_MARKERS_HALF} or {N_MARKERS_FULL} markers "
        f"(12 or 24 coordinates) but got {nMarkers} markers."
    )
    raise ValidationError(msg)


def normalise_data(markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Centre the marker data by subtracting the average shape.

    Args:
        markers: Marker data array (n_frames, n_coords*n_markers)

    Returns:
        Tuple of (normalized_markers, average_shape)

    Raises:
        ShapeError: If marker data shape is invalid
    """
    if len(markers.shape) == N_SPATIAL_DIMS:
        nMarkers = markers.shape[1]
        average_shape = get_average_shape(nMarkers)
        return markers - average_shape, average_shape

    if len(markers.shape) == N_SPATIAL_DIMS - 1:
        nMarkers = markers.shape[1] // N_SPATIAL_DIMS
        average_shape = get_average_shape(nMarkers)
        return markers - average_shape.reshape(1, -1), average_shape

    msg = f"Expected 2D or 3D marker data but got shape {markers.shape}"
    raise ShapeError(msg)


def add_average_shape(data: np.ndarray, average_shape: np.ndarray) -> np.ndarray:
    """
    Add the average shape back to centered data. Typically used during reconstruction.

    Args:
        data: Centered data (e.g., DMD reconstruction), shape (n_frames, n_coords)
        average_shape: The mean shape (1, n_coords)

    Returns:
        Data with the mean shape added back
    """
    return data + average_shape


def reshape_data(
    data: np.ndarray, n_frames: int, n_markers: int, n_dims: int = 3
) -> np.ndarray:
    """
    Reshape flattened marker data into (frames, markers, dimensions).

    Args:
        data: Input data array. Needs to have n_frames * n_markers * n_dims elements
        n_frames: Number of time frames. Use -1 to infer automatically if possible
        n_markers: Number of markers
        n_dims: Number of spatial dimensions (typically 3 for x, y, z)

    Returns:
        Reshaped data array

    Raises:
        ShapeError: If data cannot be reshaped to requested dimensions
    """
    try:
        return data.reshape(n_frames, n_markers, n_dims)
    except ValueError as e:
        msg = (
            f"Cannot reshape data of shape {data.shape} to "
            f"({n_frames}, {n_markers}, {n_dims}): {e!s}"
        )
        raise ShapeError(msg) from e


def spline_interpolation(
    times: np.ndarray, markers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform cubic spline interpolation if time steps are uneven.

    Args:
        times: Array of time points
        markers: Array of marker positions corresponding to time points

    Returns:
        Tuple of (new_times, new_markers) where new_times is evenly spaced
        and new_markers contains interpolated positions
    """
    # Create new time array with no missing points
    new_times = np.linspace(times[0], times[-1], num=len(times))

    # Perform cubic spline interpolation for each marker coordinate
    new_markers = np.zeros((len(new_times), markers.shape[1]))
    for i in range(markers.shape[1]):
        cs = CubicSpline(times, markers[:, i])
        new_markers[:, i] = cs(new_times)

    return new_times, new_markers


def bin_dataframe_means(
    dataframe: pd.DataFrame, x_axis: str = "HorzDistance", bin_size: float = 0.01
) -> pd.DataFrame:
    """
    Bin a DataFrame along a specified axis and return mean values per bin.

    Groups the data into bins of the given size along ``x_axis``, then
    computes the mean of every numeric column in each bin.  Object columns
    are carried forward using the first value per bin.  The resulting
    DataFrame has a ``time`` column set to the bin centres and a constant
    ``seqID`` of ``"binned"``.

    Args:
        dataframe: Input DataFrame to be binned
        x_axis: Column name to use for binning (e.g., 'HorzDistance')
        bin_size: Size of each bin along the x_axis
    Returns:
        DataFrame with one row per bin centre, containing mean values
        for each numeric column.
    """
    x_min = dataframe[x_axis].min()
    x_max = dataframe[x_axis].max()
    bins = np.arange(x_min, x_max + bin_size, bin_size)
    bins = np.around(bins, 3)  # Round to avoid floating point issues
    bin_centres = bins[:-1] + bin_size / 2

    # Assign each row to a bin
    dataframe_copy = dataframe.copy()
    dataframe_copy["bin"] = pd.cut(
        dataframe_copy[x_axis], bins=bins, right=False, include_lowest=True
    )

    dataframe_copy["HorzDistance"] = dataframe_copy["HorzDistance"].astype(float)
    dataframe_copy["VertDistance"] = dataframe_copy["VertDistance"].astype(float)

    # Group by bin and calculate mean for all numeric columns
    #   -- if categorical columns are present, they will be ignored in the mean
    #       calculation
    grouped = dataframe_copy.groupby("bin", observed=True).mean(numeric_only=True)

    # For object (aka categorical) columns, take the first value in each bin
    object_cols = dataframe_copy.select_dtypes(include=["object"]).columns
    for col in object_cols:
        grouped[col] = dataframe_copy.groupby("bin", observed=True)[col].first()

    # Set the time column to the bin centres and seqID to "binned"
    grouped["time"] = bin_centres[: len(grouped)]
    grouped["seqID"] = "binned"

    return grouped


def expand_time_sequence(
    times: np.ndarray, expansion_factor: float = 3.0
) -> np.ndarray:
    """Create an expanded time sequence by interpolating between original time points.

    Args:
        times: Original time points array
        expansion_factor: How many times longer the new sequence should be
        (default: 3.0)

    Returns:
        Expanded time sequence with the same number of points as original
    """
    return np.linspace(
        times[0], times[-1] * expansion_factor, len(times) * int(expansion_factor)
    )
