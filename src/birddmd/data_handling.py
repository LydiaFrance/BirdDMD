"""
Core data loading and preprocessing functions for BirdDMD.

This module handles loading, validating, and preprocessing bird marker data.
It provides functions for data loading, shape validation, normalization,
and basic data transformations.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, Union, Dict, Any

from morphing_birds import Hawk3D

# Constants
SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'samples'))

# Error handling
class DataError(Exception):
    """Base class for data-related errors."""
    pass

class ShapeError(DataError):
    """Raised when array shapes are invalid."""
    pass

class ValidationError(DataError):
    """Raised when data validation fails."""
    pass

def validate_marker_data(data: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """Validate marker data shape and return dimensions.
    
    Args:
        data: Input data array
        
    Returns:
        Tuple of (validated_data, n_frames, n_markers, total_coords)
        
    Raises:
        ShapeError: If data shape is invalid
    """
    if len(data.shape) not in [2, 3]:
        raise ShapeError(f"Data must be 2D (n_frames, n_markers*3) or 3D (n_frames, n_markers, 3), got shape {data.shape}")
        
    if len(data.shape) == 3:
        n_frames, n_markers, n_coords = data.shape
        if n_coords != 3:
            raise ShapeError(f"Expected 3 coordinates per marker, got {n_coords}")
        data_flat = data.reshape(n_frames, -1)
        total_coords = n_markers * n_coords
    else:
        n_frames, total_coords = data.shape
        if total_coords % 3 != 0:
            raise ShapeError(f"Number of coordinates ({total_coords}) must be divisible by 3")
        n_markers = total_coords // 3
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
        raise ValidationError(f"Time vector length ({len(times)}) must match number of frames ({n_frames})")
        
    return times

def validate_average_shape(avg_shape: np.ndarray, n_coords: int, is_timeseries: bool = False) -> np.ndarray:
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
        raise ValidationError("Average shape must be provided for marker data")
        
    avg_shape = avg_shape.reshape(1, -1)
    if not is_timeseries and avg_shape.shape[1] != n_coords:
        raise ValidationError(f"Average shape coordinates ({avg_shape.shape[1]}) must match data coordinates ({n_coords})")
        
    return avg_shape


def load_bird_data(bird_name: str, 
                   behaviour: str, 
                   perch_distance: Optional[str] = None, 
                   bilateral: str = "Bilateral", 
                   turn: str = "Straight", 
                   verbose: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load bird marker and info data into a DataFrame based on specified parameters.

    Args:
        bird_name: Name of the bird (e.g., 'Toothless')
        behaviour: Flight behaviour (e.g., 'Initial')
        perch_distance: Perch distance (e.g., '9m'). If None, assumes filename doesn't specify distance
        bilateral: 'Bilateral' or 'Unilateral'
        turn: Turn direction ('Straight', 'Left', 'Right')

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
        file_path = f"{SAMPLES_DIR}/{behaviour}_{perch_distance}{turn}Turn{bird_name}_{bilateral_status}.npz"
        turn_text = f", with {turn.lower()} turns only"
    else:
        file_path = f"{SAMPLES_DIR}/{behaviour}_{perch_distance}{bird_name}_{bilateral_status}.npz"
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
            print(f"Loading {bird_name} {bilateral_status.lower()} data with {perch_distance_text} perch distances{turn_text}, {behaviour.lower()}.")
            print(f"Dataframe with shape {wingbeat_df.shape} loaded. Number of sequences: {n_sequences}")
        
        return wingbeat_df, marker_column_names

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise ValidationError(f"Error loading data: {str(e)}")


def remove_time_duplicates(df: pd.DataFrame, column_name: str = 'frameID') -> pd.DataFrame:
    """
    Removes duplicate rows based on a column (typically 'frameID').

    Args:
        df: Input DataFrame
        column_name: Column to check for duplicates

    Returns:
        DataFrame with duplicates removed and index reset
    """
    # Identify duplicates
    duplicates = df[df.duplicated(subset=[column_name], keep=False)]
    ratio_duplicates = (duplicates.shape[0] / df.shape[0]) * 100
    
    # Report percentage of duplicates
    print(f"Duplicated frames are {np.round(ratio_duplicates, 2)}% of the total data.")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[column_name], keep='first')
    
    # Reset the index
    df = df.reset_index(drop=True)
    
    return df

def load_sequence_data(df: pd.DataFrame, seqID: str, marker_column_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts marker coordinates and timestamps for a specific seqID from the main DataFrame.

    Args:
        df: DataFrame containing all sequences
        seqID: The specific sequence ID to extract
        marker_column_names: List of marker coordinate column names

    Returns:
        Tuple of (markers, times) where markers is shape (n_frames, n_coords)
        and times is shape (n_frames,)
    """
    condition = df['seqID'] == seqID
    markers = df[condition][marker_column_names].to_numpy().astype(np.float64)
    times = df[condition]['time'].to_numpy().astype(np.float64)
    return markers, times

def get_average_shape(nMarkers: int, mean_shape_path: Optional[str] = None) -> np.ndarray:
    """
    Loads the mean hawk shape from a standard file based on the number of markers.

    Args:
        nMarkers: The number of markers (4 or 8)
        mean_shape_path: Path to the mean hawk shape CSV file. If None, uses default path

    Returns:
        The mean shape as a (1, n_coords) NumPy array

    Raises:
        ValidationError: If number of markers is invalid
    """
    if mean_shape_path is None:
        mean_shape_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mean_hawk_shape.csv')
    
    hawk3d = Hawk3D(mean_shape_path)
    if nMarkers == 8:
        return hawk3d.markers
    elif nMarkers == 4:
        return hawk3d.right_markers
    else:
        raise ValidationError(f"Expected 4 markers or 8 (12 or 24 coordinates) but got {nMarkers} markers.")

def normalise_data(markers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center the marker data by subtracting the average shape.

    Args:
        markers: Marker data array (n_frames, n_coords*n_markers)

    Returns:
        Tuple of (normalized_markers, average_shape)

    Raises:
        ShapeError: If marker data shape is invalid
    """
    if len(markers.shape) == 3:
        nMarkers = markers.shape[1]
        average_shape = get_average_shape(nMarkers)
        return markers - average_shape, average_shape

    elif len(markers.shape) == 2:
        nMarkers = markers.shape[1]//3
        average_shape = get_average_shape(nMarkers)
        return markers - average_shape.reshape(1, -1), average_shape

    else:
        raise ShapeError(f"Expected 2D or 3D marker data but got shape {markers.shape}")

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

def reshape_data(data: np.ndarray, n_frames: int, n_markers: int, n_dims: int = 3) -> np.ndarray:
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
        raise ShapeError(f"Cannot reshape data of shape {data.shape} to ({n_frames}, {n_markers}, {n_dims}): {str(e)}")

def spline_interpolation(times: np.ndarray, markers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
