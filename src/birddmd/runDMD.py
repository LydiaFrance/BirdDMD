import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, Union, Dict, Any

from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing

import warnings
warnings.filterwarnings("ignore")

from morphing_birds import Hawk3D

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'samples'))

def validate_marker_data(data: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """Validate marker data shape and return dimensions.
    
    Args:
        data: Input data array
        
    Returns:
        Tuple of (validated_data, n_frames, n_markers, total_coords)
    """
    if len(data.shape) not in [2, 3]:
        raise ValueError(f"Data must be 2D (n_frames, n_markers*3) or 3D (n_frames, n_markers, 3), got shape {data.shape}")
        
    if len(data.shape) == 3:
        n_frames, n_markers, n_coords = data.shape
        if n_coords != 3:
            raise ValueError(f"Expected 3 coordinates per marker, got {n_coords}")
        data_flat = data.reshape(n_frames, -1)
        total_coords = n_markers * n_coords
    else:
        n_frames, total_coords = data.shape
        if total_coords % 3 != 0:
            raise ValueError(f"Number of coordinates ({total_coords}) must be divisible by 3")
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
    """
    if times is None:
        return np.arange(n_frames)
        
    if len(times) != n_frames:
        raise ValueError(f"Time vector length ({len(times)}) must match number of frames ({n_frames})")
        
    return times

def validate_average_shape(avg_shape: np.ndarray, n_coords: int, is_timeseries: bool = False) -> np.ndarray:
    """Validate average shape dimensions.
    
    Args:
        avg_shape: Average shape array
        n_coords: Expected number of coordinates (total coordinates for marker data)
        is_timeseries: Whether this is time series data (not marker data)
        
    Returns:
        Validated average shape
    """
    if avg_shape is None:
        if is_timeseries:
            return np.zeros((1, n_coords))
        raise ValueError("Average shape must be provided for marker data")
        
    avg_shape = avg_shape.reshape(1, -1)
    if not is_timeseries and avg_shape.shape[1] != n_coords:
        raise ValueError(f"Average shape coordinates ({avg_shape.shape[1]}) must match data coordinates ({n_coords})")
        
    return avg_shape

def run_sequence_dmd(bird_name, perch_dist, turn, behaviour, seqID, 
                    n_modes=10, d=2, eig_constraints={"conjugate_pairs"},
                    min_seq_length=None, interpolate=False, verbose=True):
    """
    Run DMD analysis on a specific sequence from the dataset.
    This function handles loading the data and running DMD analysis.

    Parameters:
        bird_name: Name of the bird
        perch_dist: Perch distance
        turn: Turn direction
        behaviour: Flight behaviour
        seqID: Sequence ID
        n_modes: Number of DMD modes
        d: Hankel matrix delay parameter
        eig_constraints: Constraints for BOPDMD eigenvalues
        min_seq_length: Minimum sequence length required
        interpolate: Whether to interpolate missing time points
        verbose: Whether to print status messages

    Returns:
        tuple: (times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints)
    """
    # Load data
    df, marker_column_names = load_bird_data(
        bird_name=bird_name, 
        behaviour=behaviour, 
        perch_distance=perch_dist,
        turn=turn
    )

    if verbose:
        print(f"Running DMD for {bird_name}, {perch_dist}m, {turn} turn, seqID: {seqID}")

    # Get average shape
    n_markers = len(marker_column_names)//3
    average_shape = get_average_shape(n_markers)

    # Run DMD on sequence
    return run_DMD_sequence(
        seqID=seqID,
        df=df,
        marker_column_names=marker_column_names,
        average_shape=average_shape,
        n_Modes=n_modes,
        min_seq_length=min_seq_length,
        eig_constraints=eig_constraints,
        d=d,
        interpolate=interpolate
    )

def run_marker_dmd(markers, times=None, average_shape=None, n_modes=10, d=2,
                  eig_constraints={"conjugate_pairs"}, verbose=True):
    """
    Run DMD analysis directly on marker data.
    This function assumes the data is already loaded and preprocessed.

    Parameters:
        markers: Marker data array, shape (n_frames, n_markers, 3)
        times: Optional array of timestamps
        average_shape: Mean shape to subtract/add back
        n_modes: Number of DMD modes
        d: Hankel matrix delay parameter
        eig_constraints: Constraints for BOPDMD eigenvalues
        verbose: Whether to print status messages

    Returns:
        tuple: (Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction)
    """
    if len(markers.shape) != 3:
        raise ValueError("Markers must be 3D array (n_frames, n_markers, 3)")

    n_markers = markers.shape[1]
    if average_shape is None:
        average_shape = get_average_shape(n_markers)

    return run_dmd(
        data=markers,
        times=times,
        n_modes=n_modes,
        d=d,
        eig_constraints=eig_constraints,
        average_shape=average_shape,
        num_markers=n_markers,
        verbose=verbose
    )

def run_timeseries_dmd(data, times=None, n_modes=10, d=2,
                      eig_constraints={"conjugate_pairs"}, verbose=True):
    """
    Run DMD analysis on generic time series data.
    This function handles any 2D time series data (e.g., PCA scores).

    Parameters:
        data: Time series data array, shape (n_timesteps, n_variables)
        times: Optional array of timestamps
        n_modes: Number of DMD modes
        d: Hankel matrix delay parameter
        eig_constraints: Constraints for BOPDMD eigenvalues
        verbose: Whether to print status messages

    Returns:
        tuple: (Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction)
    """
    if len(data.shape) != 2:
        raise ValueError("Data must be 2D array (n_timesteps, n_variables)")

    return run_dmd(
        data=data,
        times=times,
        n_modes=n_modes,
        d=d,
        eig_constraints=eig_constraints,
        verbose=verbose
    )

# Update run_single_wingbeat_dmd to use the new specialized function
def run_single_wingbeat_dmd(bird_name, perch_dist, turn="Straight", behaviour="Initial",
                          seqID=None, n_modes=6, d=2, eig_constraints={"conjugate_pairs"},
                          min_seq_length=None, interpolate=False, verbose=True):
    """
    Legacy function that maintains backward compatibility.
    Now uses run_sequence_dmd internally.
    """
    return run_sequence_dmd(
        bird_name=bird_name,
        perch_dist=perch_dist,
        turn=turn,
        behaviour=behaviour,
        seqID=seqID,
        n_modes=n_modes,
        d=d,
        eig_constraints=eig_constraints,
        min_seq_length=min_seq_length,
        interpolate=interpolate,
        verbose=verbose
    )

# ------------- Loop DMD every Sequence ------------- 

def dmd_loop_seqs(bird_name,  
                  save_path,
                  behaviour,
                  perch_distance=None, 
                  bilateral="Bilateral", 
                  turn = "Straight", 
                  n_Modes=6, 
                  min_seq_length= None,
                  eig_constraints={"conjugate_pairs"}, 
                  d=2):

    """
    Loops through all unique sequences in a dataset, runs DMD, and saves results.

    Parameters:
        bird_name: Name of the bird dataset.
        save_path: Directory path to save the DMD results (.npz files).
        behaviour: Flight behaviour description.
        perch_distance: Perch distance condition. If None, uses all distances.
        bilateral: Data type ('Bilateral' or 'Unilateral').
        turn: Turn condition.
        n_Modes: Number of DMD modes.
        min_seq_length: Minimum sequence length to process.
        eig_constraints: Constraints for BOPDMD eigenvalues.
        d: Hankel matrix delay parameter.
    """
    df, marker_column_names = load_bird_data(bird_name, behaviour, perch_distance, bilateral, turn)

    seqID_counts = df['seqID'].value_counts()

    # Create a histogram
    plt.figure(figsize=(3,3))
    plt.hist(seqID_counts, bins=20)
    plt.xlabel("Number of frames per sequence")
    plt.show()


    nMarkers=len(marker_column_names)//3
    average_shape = get_average_shape(nMarkers)

    df = remove_time_duplicates(df)

    for seqID in df['seqID'].unique():

        times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_DMD_sequence(seqID, df, marker_column_names, average_shape, n_Modes, min_seq_length, eig_constraints, d)
        # TODO 2024-07-08 21:41 Maybe fix this, split into two steps to check size
        if times is not None:
            save_sequence_results(save_path, 
                                seqID, times, markers, 
                                Lambda, Modes, bn, Psi, keypoints)



# ====================================================
#              Core DMD Sequence Processing
# ====================================================

def run_DMD_sequence(seqID:str, 
                     df, 
                     marker_column_names, 
                     average_shape=None, 
                     n_Modes:int=6,
                     min_seq_length= None, 
                     eig_constraints={"conjugate_pairs"}, 
                     d:int=2, 
                     interpolate: bool = False):
    """
    Runs the complete DMD analysis pipeline for a single sequence.

    Parameters:
        seqID: The ID of the sequence to process.
        df: DataFrame containing the full dataset.
        marker_column_names: List of column names corresponding to marker coordinates.
        average_shape: The mean shape to subtract/add back. If None, it's calculated.
        n_Modes: Number of DMD modes.
        min_seq_length: Minimum sequence length required.
        eig_constraints: Constraints for BOPDMD eigenvalues.
        d: Hankel matrix delay parameter.
        interpolate: Whether to interpolate missing time points.

    Returns:
        tuple: Contains times, original_markers_reshaped, Lambda (imaginary eigenvalues),
               Modes (magnitudes), bn (amplitudes), Psi (complex modes), phase_shifts,
               keypoints (reconstructed trajectory). Returns None for all if sequence is too short.
    """
    markers, times = load_sequence_data(df, seqID, marker_column_names)

    if average_shape is None:
        nMarkers = len(marker_column_names)//3
        average_shape = get_average_shape(nMarkers)
    else:
        num_markers = average_shape.shape[1]

    if min_seq_length == None:
        min_seq_length = n_Modes+1

    if markers.shape[0] <= min_seq_length:
        return None, None, None, None, None, None, None, None, None

    # Perform spline interpolation to fill in missing time points
    if interpolate:
        times, markers = spline_interpolation(times, markers)

    # Reshape markers to (n_frames, n_markers, 3)
    markers = markers.reshape(-1, num_markers, 3)

    # Use the unified run_dmd function
    Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_dmd(
        markers,
        times=times,
        n_modes=n_Modes,
        d=d,
        eig_constraints=eig_constraints,
        average_shape=average_shape,
        num_markers=num_markers,
        verbose=False
    )

    if dmd_results is None:
        return None, None, None, None, None, None, None, None, None

    return times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints

def _compute_dmd(data, times, n_modes, eig_constraints={"conjugate_pairs"}, d=2):
    """
    Low-level function that performs the core DMD computation using pydmd's BOPDMD.
    This is an internal implementation detail and should not be called directly.
    Use run_dmd() instead.

    Parameters:
        data: Input data in shape (n_variables, n_timesteps)
        times: Timestamps corresponding to data
        n_modes: Target rank (number of modes) for DMD
        eig_constraints: Constraints applied to eigenvalues during optimization
        d: Delay embedding parameter for Hankel matrix

    Returns:
        dmd_results: Fitted pydmd BOPDMD object
        forecast: Reconstructed trajectory using dmd_results.forecast
    """
    # Debug prints
    print("\nDMD Execution Debug:")
    print(f"Data shape: {data.shape}")
    print(f"Times shape: {times.shape}")
    print(f"Number of modes: {n_modes}")
    print(f"Eigenvalue constraints: {eig_constraints}")
    print(f"Delay parameter d: {d}")
    print(f"Data min/max: {np.min(data):.3f}/{np.max(data):.3f}")
    print(f"Times min/max: {np.min(times):.3f}/{np.max(times):.3f}")

    # Create and configure DMD model
    dmd_results = hankel_preprocessing(
        BOPDMD(svd_rank=n_modes, eig_constraints=eig_constraints),
        d=d
    )

    print(f"Time size: {times.shape}")
    print(f"Data shape: {data.shape}")

    # Ensure data shape matches timestep length
    if data.shape[1] != times.shape[0]:
        if data.shape[0] == times.shape[0]:
            data = data.T
        else:
            raise ValueError("Data and times shapes do not match")

    print("\nFitting DMD...")
    dmd_results.fit(data, t=times[1:])
    print("DMD fit complete")
    print(f"Number of eigenvalues: {len(dmd_results.eigs)}")
    print(f"Eigenvalues: {np.round(dmd_results.eigs, 3)}")
    
    forecast = dmd_results.forecast(times[1:])
    return dmd_results, forecast

def run_dmd(data: np.ndarray, 
            times: Optional[np.ndarray] = None,
            n_modes: int = 10,
            d: int = 2,
            eig_constraints: set = {"conjugate_pairs"},
            average_shape: Optional[np.ndarray] = None,
            num_markers: Optional[int] = None,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]:
    """Main public interface for running DMD analysis on time series data.
    
    This function handles both marker data (3D coordinates) and generic time series data
    with robust shape validation and transformation.
    
    Args:
        data: Input data array. Can be:
              - Marker data: shape (n_frames, n_markers, 3) or (n_frames, n_markers*3)
              - Generic time series: shape (n_variables, n_timesteps)
        times: Optional array of timestamps
        n_modes: Number of DMD modes to compute
        d: Hankel matrix delay parameter
        eig_constraints: Constraints for BOPDMD eigenvalues
        average_shape: Mean shape for marker data
        num_markers: Number of markers (required for marker data)
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction)
    """
    # Check if this is time series data (n_variables, n_timesteps)
    is_timeseries = len(data.shape) == 2 and data.shape[0] < data.shape[1]
    
    if is_timeseries:
        # For time series data, transpose to (n_variables, n_timesteps)
        data_flat = data.T if data.shape[0] > data.shape[1] else data
        n_frames = data_flat.shape[1]
        n_coords = data_flat.shape[0]
        n_markers = None
    else:
        # For marker data, validate and transform
        data_flat, n_frames, n_markers, n_coords = validate_marker_data(data)
    
    # Validate times
    times = validate_times(times, n_frames)
    
    # For marker data, validate average shape
    if not is_timeseries:
        if num_markers is None:
            raise ValueError("num_markers must be provided for marker data")
        average_shape = validate_average_shape(average_shape, n_coords, is_timeseries=False)
    
    if verbose:
        print(f"Running DMD with {n_modes} modes, delay d={d}")
        print(f"Input shape: {data_flat.shape}")
    
    # Normalize data if it's marker data
    if not is_timeseries:
        data_flat, _ = normalise_data(data_flat)
    
    # Perform DMD computation
    dmd_results, _ = _compute_dmd(data_flat, times, n_modes, eig_constraints=eig_constraints, d=d)
    
    if dmd_results is None:
        if verbose:
            print("DMD fit failed")
        return None, None, None, None, None, None, None
    
    # Get the actual number of variables in the DMD results
    n_vars_dmd = dmd_results.modes.shape[0]
    if verbose:
        print(f"Number of variables in DMD results: {n_vars_dmd}")
    
    # Reorder results
    Lambda, Modes, bn, Psi, phase_shifts = reorder_dmd_results(
        dmd_results,
        num_markers if num_markers is not None else n_vars_dmd,
        n_modes,
        reshape_modes=(num_markers is not None)
    )
    
    if verbose:
        print(f"Complex Eigenvalues (log): {np.round(dmd_results.eigs, 3)}")
        print(f"Number of modes in Psi: {Psi.shape[1]}")
    
    # Reconstruct time series
    reconstruction = reconstruct_dmd(
        times,
        np.imag(dmd_results.eigs),
        dmd_results.modes[:n_coords,:],
        dmd_results.amplitudes
    )
    
    # If this is marker data, reshape the reconstruction
    if not is_timeseries:
        reconstruction = reconstruction.reshape(-1, num_markers, 3)
        if average_shape is not None:
            reconstruction = reconstruction + average_shape.reshape(-1, num_markers, 3)
    
    return Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction

# ====================================================
#         Helper Functions: Data Handling & Preprocessing
# ====================================================

def spline_interpolation(times, markers):
    """
    Perform cubic spline interpolation if time steps are uneven.

    Parameters:
        times: Array of time points.
        markers: Array of marker positions corresponding to time points.

    Returns:
        new_times: Array of evenly spaced time points.
        new_markers: Array of interpolated marker positions.
    """
    # Create new time array with no missing points
    new_times = np.linspace(times[0], times[-1], num=len(times))

    # Perform cubic spline interpolation for each marker coordinate
    new_markers = np.zeros((len(new_times), markers.shape[1]))
    for i in range(markers.shape[1]):
        cs = CubicSpline(times, markers[:, i])
        new_markers[:, i] = cs(new_times)

    return new_times, new_markers

# ------------- Loading and Preprocessing ------------- 

def load_bird_data(bird_name, 
                   behaviour, 
                   perch_distance=None, 
                   bilateral="Bilateral", 
                   turn = "Straight"):
    """
    Load bird marker and info data into a DataFrame based on specified parameters.

    Constructs the filename based on conventions and loads .npz data.

    Parameters:
        bird_name: Name of the bird (e.g., 'Toothless').
        behaviour: Flight behaviour (e.g., 'Initial').
        perch_distance: Perch distance (e.g., '9m'). If None, assumes filename doesn't specify distance.
        bilateral: 'Bilateral' or 'Unilateral'.
        turn: Turn direction ('Straight', 'Left', 'Right'). Used for specific filenames (e.g., 9m).

    Returns:
        wingbeat_df: Combined DataFrame of info and marker data.
        marker_column_names: List of marker coordinate column names.
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
    print(f"Loading {bird_name} {bilateral_status.lower()} data with {perch_distance_text} perch distances{turn_text}, {behaviour.lower()}.")
    print(f"Dataframe with shape {wingbeat_df.shape} loaded. Number of sequences: {n_sequences}")
    
    return wingbeat_df, marker_column_names


def remove_time_duplicates(df, column_name='frameID'):
    """
    Removes duplicate rows based on a column (typically 'frameID').

    Reports the percentage of duplicates found before removal.

    Parameters:
        df: Input DataFrame.
        column_name: Column to check for duplicates.

    Returns:
        df: DataFrame with duplicates removed and index reset.
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


# ------------- Running DMD ------------- 


def load_sequence_data(df, seqID, marker_column_names):
    """
    Extracts marker coordinates and timestamps for a specific seqID from the main DataFrame.

    Parameters:
        df: DataFrame containing all sequences.
        seqID: The specific sequence ID to extract.
        marker_column_names: List of marker coordinate column names.

    Returns:
        markers: NumPy array of marker data for the sequence (n_frames, n_coords).
        times: NumPy array of timestamps for the sequence.
    """

    condition = df['seqID'] == seqID
    markers = df[condition][marker_column_names].to_numpy().astype(np.float64)
    times = df[condition]['time'].to_numpy().astype(np.float64)
    return markers, times

def get_average_shape(nMarkers, mean_shape_path=None):
    """
    Loads the mean hawk shape from a standard file based on the number of markers.

    Parameters:
        nMarkers: The number of markers (4 or 8).
        mean_shape_path: Path to the mean hawk shape CSV file. If None, uses default path.

    Returns:
        average_shape: The mean shape as a (1, n_coords) NumPy array.
        nMarkers: The number of markers (4 or 8).
    """
    if mean_shape_path is None:
        mean_shape_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mean_hawk_shape.csv')
    
    hawk3d = Hawk3D(mean_shape_path)
    if nMarkers == 8:
        return hawk3d.markers
    elif nMarkers == 4:
        return hawk3d.right_markers
    else:
        raise ValueError("Expected 4 markers or 8 (12 or 24 coordinates) but got {nMarkers} markers.")
    

def normalise_data(markers):
    """
    Center the marker data by subtracting the average shape.

    Parameters:
        markers: Marker data array (n_frames, n_coords*n_markers).
        average_shape: The mean shape (1, n_coords*n_markers).

    Returns:
        normalized_markers: Centered marker data.
        average_shape: The average shape passed in (for convenience).
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
        raise ValueError("Expected 2D or 3D marker data but got shape", markers.shape)
    





def add_average_shape(data, average_shape):
    """
    Add the average shape back to centered data. Typically used during reconstruction.

    Parameters:
        data: Centered data (e.g., DMD reconstruction), shape (n_frames, n_coords).
        average_shape: The mean shape (1, n_coords).

    Returns:
        Data with the mean shape added back.
    """
    return data + average_shape


def reshape_data(data, n_frames, n_markers, n_dims=3):
    """
    Reshape flattened marker data into (frames, markers, dimensions).

    Parameters:
        data: Input data array. Needs to have n_frames * n_markers * n_dims elements.
        n_frames: Number of time frames. Use -1 to infer automatically if possible.
        n_markers: Number of markers.
        n_dims: Number of spatial dimensions (typically 3 for x, y, z).

    Returns:
        Reshaped data array.
    """
    return data.reshape(n_frames, n_markers, n_dims)


def run_forecast(dmd_results, times, average_shape, num_markers):
    """
    Reconstructs the full trajectory using all computed DMD modes.

    Uses the `reconstruct_dmd` function based on phasor notation and adds back the mean shape.

    Parameters:
        dmd_results: Fitted pydmd object containing modes, eigenvalues, amplitudes.
        times: Time vector for reconstruction.
        average_shape: Mean shape to add back.
        num_markers: Number of markers.

    Returns:
        keypoints: Reconstructed trajectory reshaped to (n_frames, n_markers, 3).
                   # Frame adjustment note removed for clarity, handled below if needed
    """
    n_coords = num_markers*3
    print(f"n_coords: {n_coords}")
    # Pass the imaginary part of eigenvalues as omega (frequencies)
    reconstruction = reconstruct_dmd(times,
                                   np.imag(dmd_results.eigs), # Pass frequencies
                                   dmd_results.modes[:n_coords,:],
                                   dmd_results.amplitudes)

    # Add mean shape and reshape
    # Ensure reconstruction shape is (n_times, n_coords) before adding mean shape
    average_shape = average_shape.reshape(1, -1)
    if reconstruction.shape[1] != average_shape.shape[1]:
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Average shape shape: {average_shape.shape}")
        raise ValueError(f"Shape mismatch: Reconstruction ({reconstruction.shape}) vs Average Shape ({average_shape.shape})")
    
    
    forecast_plus_mean = reconstruction + average_shape # Broadcasting
    keypoints = reshape_data(forecast_plus_mean, -1, num_markers, 3) # Use reshape_data

    # Frame adjustment: This might be needed if dmd.fit used times[1:]
    # Check if the length matches the original marker data length.
    # If `perform_dmd` fits on `times[1:]`, the reconstruction using `times` might
    # naturally align or might need adjustment depending on pydmd's forecast behavior.
    # Let's assume reconstruction length matches input `times` length for now.
    # If an adjustment is needed, uncommenting the line below might be correct:
    # keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)

    return keypoints


def reorder_dmd_results(dmd_results, num_markers, nModes, order_by='amplitude', reshape_modes=True):
    """
    Reorders DMD results by either amplitude or frequency magnitude.

    Parameters:
        dmd_results: Fitted pydmd object.
        num_markers: Number of markers (for marker data) or number of variables (for time series).
        nModes: Number of modes computed.
        order_by: String indicating ordering method ('amplitude' or 'frequency').
        reshape_modes: Whether to reshape modes into (markers, coordinates, modes) format.
                      Only applies to marker data.

    Returns:
        Lambda_imag: Imaginary part of eigenvalues (frequencies), sorted.
        Modes_mag: Magnitudes of modes, sorted.
        bn: Amplitudes, sorted.
        Psi: Complex modes, sorted.
        phase_shifts: Phase shifts of modes in radians, sorted.
    """
    if order_by not in ['amplitude', 'frequency']:
        raise ValueError("order_by must be either 'amplitude' or 'frequency'")

    # Get frequencies for either sorting or phase calculation
    frequencies = np.imag(dmd_results.eigs)
    sign_omega = np.sign(frequencies)

    # Determine sort order based on method
    if order_by == 'amplitude':
        sort_idx = np.argsort(np.abs(dmd_results.amplitudes))[::-1]
    else:  # frequency
        sort_idx = np.argsort(np.abs(frequencies))[::-1]
    
    # Get modes and reorder
    Psi = dmd_results.modes
    
    # For time series data, take only the first n_vars rows
    # For marker data, take all coordinates
    n_vars = num_markers if not reshape_modes else num_markers * 3
    Psi = Psi[:n_vars, sort_idx]
    
    # Handle marker data vs time series data
    if reshape_modes and num_markers <= 8:  # Marker data case
        mode_magnitudes = np.sqrt(np.real(Psi)**2 + np.imag(Psi)**2)
        Modes = mode_magnitudes.reshape(num_markers, 3, nModes)
    else:  # Time series case
        mode_magnitudes = np.sqrt(np.real(Psi)**2 + np.imag(Psi)**2)
        Modes = mode_magnitudes
    
    # Calculate phase shifts
    phase_shifts = np.arctan2(-sign_omega[sort_idx] * np.imag(Psi), np.real(Psi))

    # Get eigenvalues and reorder
    Lambda = dmd_results.eigs[sort_idx]

    return np.imag(Lambda), Modes, dmd_results.amplitudes[sort_idx], Psi, phase_shifts



def plot_amplitude_ranking(
    markers,
    times,
    max_modes=20,
    d=2,
    eig_constraints={"conjugate_pairs"}, # Default to notebook example
    figsize=(6, 2)
):
    """
    Runs DMD with a specified maximum number of modes and plots the
    sorted absolute amplitudes to help determine an appropriate rank.

    This is useful for visually inspecting the decay of mode amplitudes
    to choose the `n_modes` parameter for main DMD runs.

    Parameters:
        markers: Marker data
        times: Time vector corresponding to markers.
        max_modes: The maximum rank (number of modes) to compute.
        d: Hankel matrix delay parameter.
        eig_constraints: Constraints for BOPDMD eigenvalues.
        figsize: Size of the output plot.

    Returns:
        fig: Matplotlib figure object.
        ax: Matplotlib axes object.
        sorted_amplitudes: NumPy array of sorted absolute amplitudes (descending).
    """
    # Input validation
    if times.shape[0] != markers.shape[0]:
        raise ValueError("Shape mismatch: `times` and `markers` must have the same number of frames.")
    # Check if enough data points exist for the requested modes and delay
    required_steps = max_modes + d # Minimum steps needed for Hankel matrix construction and SVD
    if markers.shape[0] <= required_steps:
        raise ValueError(f"Not enough time steps ({markers.shape[0]}) to compute "
                         f"{max_modes} modes with delay d={d}. Need > {required_steps}.")

    # Transpose data for DMD fit (coords, time)
    normalised_markers, _ = normalise_data(markers)
    reshaped_markers = normalised_markers.reshape(markers.shape[0], -1)
    transposed_markers = reshaped_markers.T

    # Setup and fit DMD model
    print(f"Fitting DMD with max_modes={max_modes}, d={d}...")
    dmd_model = hankel_preprocessing(
        BOPDMD(svd_rank=max_modes, eig_constraints=eig_constraints),
        d=d
    )
    # Use times[1:] for fitting, consistent with other functions
    dmd_model.fit(transposed_markers, t=times[1:])
    print("DMD fit complete.")

    # Get computed amplitudes (complex values)
    amplitudes = dmd_model.amplitudes

    if amplitudes is None or len(amplitudes) == 0:
         print("Warning: DMD fit did not produce amplitudes.")
         # Create an empty plot or handle as needed
         fig, ax = plt.subplots(figsize=figsize)
         ax.text(0.5, 0.5, 'No amplitudes found', ha='center', va='center')
         return fig, ax, np.array([])

    # Calculate absolute amplitudes and sort descending
    abs_amplitudes = np.abs(amplitudes)
    # Argsort returns indices for ascending order, use [::-1] for descending
    sort_indices = np.argsort(abs_amplitudes)[::-1]
    sorted_amplitudes = abs_amplitudes[sort_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    # Ranks should go from 0 to number of modes found - 1
    ranks = np.arange(len(sorted_amplitudes))
    ax.scatter(ranks, sorted_amplitudes, marker='o', s=15) # Adjust marker/size
    ax.set_ylabel(r"Amplitude $|\beta|$") # Use standard notation if preferred
    ax.set_xlabel("DMD Mode Rank (Sorted by Amplitude)")
    ax.set_title(f"DMD Mode Amplitude Ranking (max_modes={max_modes})")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=8)
    # Optional: set xlim to show all potential ranks up to max_modes?
    # ax.set_xlim(-0.5, max_modes - 0.5)

    fig.tight_layout()

    return fig, ax, sorted_amplitudes


# ------------- Reconstructing Modes ------------- 


def reconstruct_dmd(times: np.ndarray,
                   omega: np.ndarray,
                   Psi: np.ndarray,
                   amplitudes: np.ndarray,
                   mode_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Reconstructs data using phasor notation for DMD modes.
    
    Args:
        times: Time vector for reconstruction
        omega: Array of frequencies (imaginary parts of eigenvalues)
        Psi: Complex DMD modes
        amplitudes: Complex DMD amplitudes
        mode_indices: Optional array of indices specifying which modes to include
        
    Returns:
        reconstruction: Real-valued reconstructed data (n_times, n_coords)
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [times, omega, Psi, amplitudes]):
        raise TypeError("All inputs must be numpy arrays")
        
    n_coords = Psi.shape[0]
    n_modes = len(omega)
    
    if len(amplitudes) != n_modes:
        raise ValueError(f"Number of amplitudes ({len(amplitudes)}) must match number of modes ({n_modes})")
        
    if Psi.shape[1] != n_modes:
        raise ValueError(f"Number of modes in Psi ({Psi.shape[1]}) must match number of frequencies ({n_modes})")
    
    # Initialize reconstruction array
    reconstruction = np.zeros((len(times), n_coords), dtype=float)
    
    # Sort by amplitude magnitude to pair modes correctly
    sort_indices = np.argsort(np.abs(amplitudes))[::-1]
    omega_sorted = omega[sort_indices]
    Psi_sorted = Psi[:, sort_indices]
    amplitudes_sorted = amplitudes[sort_indices]
    
    # Validate mode_indices if provided
    if mode_indices is not None:
        mode_indices = np.asarray(mode_indices)
        if not np.all((mode_indices >= 0) & (mode_indices < n_modes)):
            raise ValueError(f"mode_indices must be in range [0, {n_modes-1}]")
        # Convert mode_indices to the sorted space
        mode_indices = sort_indices[mode_indices]
    
    processed_indices = set()
    
    for i in range(n_modes):
        if i in processed_indices:
            continue
            
        # Skip this mode if mode_indices is provided and i is not in the sorted indices
        if mode_indices is not None and i not in mode_indices:
            continue
            
        # Find conjugate pair (or itself if real eigenvalue/zero frequency)
        target_omega = -omega_sorted[i]
        conjugate_idx = -1
        min_diff = 1e-6  # Tolerance for finding conjugate pair frequency
        
        # Efficiently find potential conjugate index
        if i % 2 == 0 and i + 1 < n_modes:
            if np.abs(omega_sorted[i+1] - target_omega) < min_diff:
                conjugate_idx = i + 1
                
        # Fallback search if not found immediately
        if conjugate_idx == -1:
            for k in range(i + 1, n_modes):
                if k not in processed_indices and np.abs(omega_sorted[k] - target_omega) < min_diff:
                    conjugate_idx = k
                    break
            # If still no conjugate found, treat as non-paired mode
            if conjugate_idx == -1:
                conjugate_idx = i
                
        # Calculate magnitude and phase for the primary mode
        sign_omega = np.sign(omega_sorted[i]) if np.abs(omega_sorted[i]) > 1e-9 else 1
        magnitude = np.abs(Psi_sorted[:, i])
        phase = np.arctan2(-sign_omega * np.imag(Psi_sorted[:, i]), np.real(Psi_sorted[:, i]))
        
        # Amplitude and frequency
        beta = np.abs(amplitudes_sorted[i])
        freq = np.abs(omega_sorted[i])
        
        # Add contribution: 2 * beta * magnitude * cos(omega*t - phase)
        # If it's a zero-frequency mode paired with itself, avoid double counting
        factor = 1.0 if i == conjugate_idx else 2.0
        
        for t_idx, t in enumerate(times):
            reconstruction[t_idx, :] += factor * beta * magnitude * np.cos(freq * t - phase)
            
        # Mark both modes as processed
        processed_indices.add(i)
        processed_indices.add(conjugate_idx)
        
    return reconstruction



# def reconstruct_dmd(times, Lambda, Psi, amplitudes):
#     """Reconstruct data using phasor notation for DMD."""
#     n_modes = len(Lambda)
#     reconstruction = np.zeros((len(times), Psi.shape[0]), dtype=complex)
    
#     for i in range(0, n_modes, 2):  # Step by 2 for conjugate pairs
#         # Calculate magnitude and phase for this mode
#         magnitude = np.sqrt(np.real(Psi[:,i])**2 + np.imag(Psi[:,i])**2)
#         phase = np.arctan2(-np.imag(Psi[:,i]), np.real(Psi[:,i]))
        
#         # Reconstruct using phasor notation
#         beta = np.abs(amplitudes[i])
#         omega = Lambda[i]
        
#         for t_idx, t in enumerate(times):
#             reconstruction[t_idx,:] += 2 * beta * magnitude * \
#                 np.cos(omega * t - phase)
    
#     return np.real(reconstruction)


def reconstruct_specific_modes(times: np.ndarray,
                             dmd_results: Any,
                             mode_indices: Union[list, np.ndarray],
                             n_markers: int = 8) -> np.ndarray:
    """Reconstruct specific DMD modes using original DMD outputs.
    
    Args:
        times: Time points array
        dmd_results: DMD object with eigs, modes, amplitudes
        mode_indices: List/array of mode indices to include
        n_markers: Number of markers (default=8)
        
    Returns:
        reconstruction: Reconstructed keypoints (n_frames, n_markers, 3)
    """
    # Input validation
    if not hasattr(dmd_results, 'modes') or not hasattr(dmd_results, 'eigs') or not hasattr(dmd_results, 'amplitudes'):
        raise ValueError("dmd_results must have modes, eigs, and amplitudes attributes")
        
    n_coords = n_markers * 3
    
    # Validate mode indices
    mode_indices = np.asarray(mode_indices)
    if not np.all((mode_indices >= 0) & (mode_indices < len(dmd_results.amplitudes))):
        raise ValueError(f"mode_indices must be in range [0, {len(dmd_results.amplitudes)-1}]")
    
    # Initialize reconstruction array
    reconstruction = np.zeros((n_coords, len(times)), dtype=complex)
    
    # Map user indices to original indices
    original_indices = np.argsort(np.abs(dmd_results.amplitudes))[::-1]
    mode_indices = original_indices[mode_indices]
    
    # Add contribution from each selected mode
    for j in mode_indices:
        # Use original (unordered) DMD components
        mode = dmd_results.modes[:n_coords, j].reshape(-1, 1)
        eig = dmd_results.eigs[j]
        amplitude = dmd_results.amplitudes[j]
        
        # Time evolution
        time_dynamics = np.exp(eig * times)
        
        # Add this mode's contribution
        reconstruction += amplitude * np.dot(mode, time_dynamics.reshape(1, -1))
    
    # Convert to real values and reshape
    reconstruction = np.real(reconstruction)
    reconstruction_transposed = reconstruction.T
    reconstruction_reshaped = reconstruction_transposed.reshape(-1, n_markers, 3)
    
    # Get average shape
    average_shape = get_average_shape(n_markers)
    
    # Add average shape and concatenate last time point
    reconstruction_keypoints = reconstruction_reshaped + average_shape
    reconstruction_keypoints = np.concatenate([reconstruction_keypoints[1:], reconstruction_keypoints[-1:]], axis=0)
    
    # Final shape validation
    expected_shape = (len(times), n_markers, 3)
    if reconstruction_keypoints.shape != expected_shape:
        raise ValueError(f"Reconstruction shape mismatch: got {reconstruction_keypoints.shape}, expected {expected_shape}")
    
    return reconstruction_keypoints

# def reconstruct_specific_modes(times, eigs_sorted, Psi_sorted, amplitudes_sorted, reordered_mode_indices):
#     """
#     Reconstructs the trajectory using only a specified subset of DMD modes,
#     based on their rank after sorting by amplitude magnitude.

#     Uses the reordered DMD components corresponding to the provided indices.

#     Parameters:
#         times: Time vector for reconstruction.
#         eigs_sorted: Complex eigenvalues, sorted by amplitude magnitude
#                      (output from reorder_dmd_results).
#         Psi_sorted: Complex mode vectors, sorted by amplitude magnitude
#                     (output from reorder_dmd_results).
#         amplitudes_sorted: Complex amplitudes, sorted by amplitude magnitude
#                            (output from reorder_dmd_results).
#         reordered_mode_indices: List or array of integer indices (0-based) specifying
#                                 which modes to use *after* they have been sorted by
#                                 amplitude. E.g., [0, 1] uses the two modes with the
#                                 highest amplitudes.

#     Returns:
#         reconstruction: Complex-valued reconstructed data from selected modes (n_coords, n_times).
#                         Take np.real() for the physical trajectory.
#     """
#     n_coords = Psi_sorted.shape[0]
#     reconstruction = np.zeros((n_coords, len(times)), dtype=complex)

#     # Check if inputs are consistent
#     if not (len(eigs_sorted) == Psi_sorted.shape[1] == len(amplitudes_sorted)):
#         raise ValueError("Input arrays (eigs, Psi, amplitudes) must have consistent number of modes.")

#     max_index = max(reordered_mode_indices) if reordered_mode_indices else -1
#     if max_index >= len(eigs_sorted):
#          raise IndexError(f"Mode index {max_index} out of bounds for {len(eigs_sorted)} available sorted modes.")

#     for j in reordered_mode_indices:
#         # Use the j-th component from the *sorted* arrays
#         mode = Psi_sorted[:, j:j+1]       # Shape (n_coords, 1)
#         eig = eigs_sorted[j]              # Scalar
#         amplitude = amplitudes_sorted[j]  # Scalar

#         # Time evolution vector for this mode
#         # Use the complex eigenvalue directly: exp(eig*t) = exp(sigma*t) * exp(j*omega*t)
#         # np.outer creates shape (1, n_times) correctly
#         time_dynamics = np.exp(np.outer(eig, times))

#         # Add this mode's contribution: amplitude * mode_vector * time_dynamics_vector
#         reconstruction += amplitude * (mode @ time_dynamics) # Matrix multiplication handles shapes

#     # Transpose the result to be (n_times, n_coords) consistent with reconstruct_dmd
#     return reconstruction.T


# ------------- Saving DMD results ------------- 

def save_sequence_results(save_path, seqID, times, markers, Lambda, Modes, bn, Psi, keypoints):
    """
    Save DMD results and metadata for a single sequence to a compressed .npz file.

    Parameters:
        save_path: Directory to save the file.
        seqID: Sequence ID, used in the filename.
        times: Time vector.
        markers: Original marker data (reshaped).
        Lambda: Imaginary eigenvalues (frequencies).
        Modes: Mode magnitudes.
        bn: Amplitudes.
        Psi: Complex modes.
        phase_shifts: Mode phase shifts.
        keypoints: Reconstructed full trajectory.
    """
    filename = os.path.join(save_path, f"{seqID}_dmd_results.npz")

    np.savez_compressed(filename, 
                        times = times,
                        markers = markers,
                        Lambda=Lambda, 
                        Modes=Modes, 
                        bn=bn, 
                        Psi=Psi, 
                        forecast_keypoints=keypoints, 
                        allow_pickle=True)


# ------------- Loading DMD results ------------- 

def load_sequence_results(save_path, seqID):
    """
    Load DMD results for a specific sequence ID from a .npz file.

    Handles FileNotFoundError gracefully.

    Parameters:
        save_path: Directory where the .npz file is located.
        seqID: Sequence ID to load.

    Returns:
        Tuple containing loaded data: (times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints).
        Returns None for all if the file is not found or phase_shifts key is missing (for older files).
    """ 
    filename = os.path.join(save_path, f"{seqID}_dmd_results.npz")
    
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return (None,) * 8 # Return tuple of Nones matching expected output count
    
    times = data.get('times')
    markers = data.get('markers')
    Lambda = data.get('Lambda')
    Modes = data.get('Modes')
    bn = data.get('bn')
    Psi = data.get('Psi')
    phase_shifts = data.get('phase_shifts')
    keypoints = data.get('forecast_keypoints')

    if times is None or Lambda is None:
         print(f"Warning: File {filename} might be missing essential data.")
    
    return times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints


def load_dmd_results(wingbeat_df, save_path):
    """
    Loads DMD results for all sequences present in a DataFrame from a specified directory.

    Iterates through unique seqIDs in the DataFrame, loads corresponding .npz files,
    and compiles the results into a new DataFrame.

    Parameters:
        wingbeat_df: DataFrame containing seqIDs to load.
        save_path: Directory containing the saved _dmd_results.npz files.

    Returns:
        DataFrame where each row corresponds to a sequence and columns contain the loaded DMD results.
    """
    # Initialize list to store data dictionaries
    DMD_dict = []

    # For each .npz file in the directory, load the numpy arrays inside and stack them
    for seqID in wingbeat_df['seqID'].unique():
        # Check if the file exists
        if not os.path.exists(f"{save_path}{seqID}_dmd_results.npz"):
            # print(f"File {save_path}{seqID}_dmd_results.npz not found.")
            continue

        times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints = load_sequence_results(save_path, seqID) # Updated variable count
        
        if times is None:
            continue

        # Append a dictionary of the data to the list
        DMD_dict.append({
            "seqID": seqID,
            "times": times,
            "markers": markers,
            "Lambda": Lambda,
            "Modes": Modes,
            "bn": bn,
            "Psi": Psi,
            "phase_shifts": phase_shifts, 
            "keypoints": keypoints
        })

        

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(DMD_dict)


# ====================================================
#       Helper Functions: PCA Projection (Post-DMD)
# ====================================================

def make_unilateral_keypoints(keypoints):
    """
    Prepares bilateral keypoint data for unilateral PCA projection.

    Separates left and right markers, mirrors the left side's x-coordinates,
    flattens the data, and concatenates left and right frames.

    Parameters:
        keypoints: Bilateral keypoint data, shape (n_frames, n_bilateral_markers, 3).
                   Assumes alternating left/right markers (L, R, L, R, ...).

    Returns:
        dmd_reconstruction_flat: Flattened data suitable for PCA (n_frames * 2, n_unilateral_markers * 3).
        left_right_bool: Boolean array indicating left (0) or right (1) side for each row in flattened data.
    """
    # Separate left and right keypoints (assuming L, R, L, R, ... structure)
    left_reconstruction = keypoints[:, ::2, :].copy()
    right_reconstruction = keypoints[:, 1::2, :].copy()

    # Mirror the x coordinate for left side to match right side's perspective
    left_reconstruction[:, :, 0] = -left_reconstruction[:, :, 0]

    # Flatten the data (frames, markers*coords)
    n_frames = keypoints.shape[0]
    n_unilateral_markers = left_reconstruction.shape[1]
    n_coords = n_unilateral_markers * 3

    left_flat = left_reconstruction.reshape(n_frames, n_coords)
    right_flat = right_reconstruction.reshape(n_frames, n_coords)

    # Concatenate left and right data vertically (stack frames)
    dmd_reconstruction_flat = np.concatenate([left_flat, right_flat], axis=0)

    # Create boolean array for left/right identification
    right_bool = np.concatenate([
        np.zeros(n_frames, dtype=bool),  # False (0) for left
        np.ones(n_frames, dtype=bool)    # True (1) for right
    ])

    return dmd_reconstruction_flat, right_bool


def project_into_pca_space(new_data: np.ndarray,
                         mu: np.ndarray,
                         principal_components: np.ndarray) -> np.ndarray:
    """Projects new shape data into an existing PCA space.
    
    Args:
        new_data: New data to project, shape (n_samples, n_features)
        mu: Mean shape used for original PCA, shape (n_features,) or (1, n_features)
        principal_components: PCA components, shape (n_components, n_features)
        
    Returns:
        scores: Projected data in PCA space, shape (n_samples, n_components)
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [new_data, mu, principal_components]):
        raise TypeError("All inputs must be numpy arrays")
        
    # Validate shapes
    n_features = principal_components.shape[1]
    if new_data.shape[1] != n_features:
        raise ValueError(f"New data features ({new_data.shape[1]}) must match PCA space ({n_features})")
        
    if mu.shape[-1] != n_features:
        raise ValueError(f"Mean shape features ({mu.shape[-1]}) must match PCA space ({n_features})")
    
    # Flatten mean shape if needed
    mu_flat = mu.reshape(-1)
    
    # Center the new data
    new_data_centered = new_data - mu_flat
    
    # Project onto principal components
    scores = np.dot(new_data_centered, principal_components.T)
    
    return scores


def project_into_coordinate_space(scores: np.ndarray,
                                mu: np.ndarray,
                                principal_components: np.ndarray,
                                n_markers: int = 8) -> np.ndarray:
    """Projects PCA scores back into original coordinate space.
    
    Args:
        scores: PCA scores to project back, shape (n_samples, n_components)
        mu: Mean shape used for original PCA, shape (n_features,) or (1, n_features)
        principal_components: PCA components, shape (n_components, n_features)
        n_markers: Number of markers in original data (default=8)
        
    Returns:
        keypoints: Reconstructed keypoints, shape (n_samples, n_markers, 3)
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [scores, mu, principal_components]):
        raise TypeError("All inputs must be numpy arrays")
        
    # Validate shapes
    n_components = principal_components.shape[0]
    n_features = principal_components.shape[1]
    
    if scores.shape[1] != n_components:
        raise ValueError(f"Scores components ({scores.shape[1]}) must match PCA components ({n_components})")
        
    if mu.shape[-1] != n_features:
        raise ValueError(f"Mean shape features ({mu.shape[-1]}) must match PCA space ({n_features})")
        
    if n_features != n_markers * 3:
        raise ValueError(f"Number of features ({n_features}) must match n_markers*3 ({n_markers*3})")
    
    # Flatten mean shape if needed
    mu_flat = mu.reshape(-1)
    
    # Project back to original space
    reconstruction = np.dot(scores, principal_components) + mu_flat
    
    # Reshape into keypoints format
    keypoints = reconstruction.reshape(-1, n_markers, 3)
    
    return keypoints

def create_scores_info_df(scores: np.ndarray,
                         seq_info_df: pd.DataFrame,
                         left_right_bool: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Combines PCA scores with sequence metadata into a DataFrame.
    
    Args:
        scores: PCA scores array, shape (n_frames_scores, n_components)
        seq_info_df: DataFrame with metadata for original sequence
        left_right_bool: Optional boolean array indicating left/right side
        
    Returns:
        combined_df: DataFrame combining scores and metadata
    """
    # Input validation
    if not isinstance(scores, np.ndarray):
        raise TypeError("scores must be a numpy array")
    if not isinstance(seq_info_df, pd.DataFrame):
        raise TypeError("seq_info_df must be a pandas DataFrame")
    if left_right_bool is not None and not isinstance(left_right_bool, np.ndarray):
        raise TypeError("left_right_bool must be a numpy array")
        
    # Get dimensions
    n_frames_scores = scores.shape[0]
    n_frames_original = len(seq_info_df)
    n_components = scores.shape[1]
    
    # Create PC column names
    pc_names = [f'PC{i:02}' for i in range(1, n_components + 1)]
    
    # Create scores DataFrame
    scores_df = pd.DataFrame(scores, columns=pc_names)
    
    # Handle metadata alignment
    if n_frames_scores == 2 * n_frames_original and left_right_bool is not None:
        # Duplicate metadata for unilateral data
        meta_df_aligned = pd.concat([seq_info_df, seq_info_df], ignore_index=True)
        meta_df_aligned['Left'] = left_right_bool.astype(int)
    elif n_frames_scores == n_frames_original:
        # Direct alignment
        meta_df_aligned = seq_info_df.copy()
        if left_right_bool is not None:
            meta_df_aligned['Left'] = left_right_bool.astype(int)
    else:
        raise ValueError(f"Score frames ({n_frames_scores}) must match original frames ({n_frames_original}) "
                       f"or be double for unilateral data")
    
    # Combine metadata and scores
    combined_df = pd.concat([meta_df_aligned, scores_df], axis=1)
    
    return combined_df

def modify_mode_frequencies(omega, Psi, amplitudes, mode_indices_to_zero=None):
    """
    Creates a copy of the DMD components with modified frequencies for specified modes.
    Useful for forcing specific modes to be constant (zero frequency) while preserving their amplitudes.

    Parameters:
        omega: Array of frequencies (imaginary parts of eigenvalues)
        Psi: Complex DMD modes
        amplitudes: Complex DMD amplitudes
        mode_indices_to_zero: List/array of mode indices to set to zero frequency.
                            If None, no modes are modified.

    Returns:
        tuple: (modified_omega, Psi, amplitudes) where modified_omega has zero frequencies
               for the specified modes while preserving conjugate pair structure.
    """
    # Make copies to avoid modifying original arrays
    modified_omega = omega.copy()
    modified_Psi = Psi.copy()
    modified_amplitudes = amplitudes.copy()
    
    if mode_indices_to_zero is None:
        return modified_omega, modified_Psi, modified_amplitudes
        
    # Sort by amplitude magnitude to ensure consistent indexing
    sort_indices = np.argsort(np.abs(amplitudes))[::-1]
    omega_sorted = modified_omega[sort_indices]
    
    # Find conjugate pairs for the modes we want to modify
    processed_indices = set()
    for i in mode_indices_to_zero:
        if i in processed_indices:
            continue
            
        # Find conjugate pair
        target_omega = -omega_sorted[i]
        conjugate_idx = -1
        min_diff = 1e-6
        
        # Look for conjugate pair
        if i % 2 == 0 and i + 1 < len(omega_sorted):
            if np.abs(omega_sorted[i+1] - target_omega) < min_diff:
                conjugate_idx = i + 1
        if conjugate_idx == -1:
            for k in range(i + 1, len(omega_sorted)):
                if k not in processed_indices and np.abs(omega_sorted[k] - target_omega) < min_diff:
                    conjugate_idx = k
                    break
                if conjugate_idx == -1:
                    conjugate_idx = i
                    
            # Set both the mode and its conjugate to zero frequency
            modified_omega[sort_indices[i]] = 0.0
            modified_omega[sort_indices[conjugate_idx]] = 0.0
            
            processed_indices.add(i)
            processed_indices.add(conjugate_idx)
    
    return modified_omega, modified_Psi, modified_amplitudes

def run_forecast_with_modified_modes(dmd_results, times, average_shape, num_markers, mode_indices_to_zero=None, selected_mode_indices=None):
    """
    Reconstructs the trajectory using selected DMD modes, with optional frequency modification.
    Can both modify frequencies of specific modes and select which modes to include.

    Parameters:
        dmd_results: Fitted pydmd object containing modes, eigenvalues, amplitudes.
        times: Time vector for reconstruction.
        average_shape: Mean shape to add back.
        num_markers: Number of markers.
        mode_indices_to_zero: List/array of mode indices to set to zero frequency.
                            If None, no frequencies are modified.
        selected_mode_indices: List/array of mode indices to include in reconstruction.
                             If None, all modes are included.

    Returns:
        keypoints: Reconstructed trajectory reshaped to (n_frames, n_markers, 3).
    """
    n_coords = num_markers*3
    
    # Get the original components
    omega = np.imag(dmd_results.eigs)
    Psi = dmd_results.modes[:n_coords,:]
    amplitudes = dmd_results.amplitudes
    
    # Sort by amplitude magnitude to ensure consistent indexing
    sort_indices = np.argsort(np.abs(amplitudes))[::-1]
    omega_sorted = omega[sort_indices]
    Psi_sorted = Psi[:, sort_indices]
    amplitudes_sorted = amplitudes[sort_indices]
    
    # Modify frequencies if requested
    if mode_indices_to_zero is not None:
        # Find conjugate pairs for the modes we want to modify
        processed_indices = set()
        for i in mode_indices_to_zero:
            if i in processed_indices:
                continue
                
            # Find conjugate pair
            target_omega = -omega_sorted[i]
            conjugate_idx = -1
            min_diff = 1e-6
            
            # Look for conjugate pair
            if i % 2 == 0 and i + 1 < len(omega_sorted):
                if np.abs(omega_sorted[i+1] - target_omega) < min_diff:
                    conjugate_idx = i + 1
            if conjugate_idx == -1:
                for k in range(i + 1, len(omega_sorted)):
                    if k not in processed_indices and np.abs(omega_sorted[k] - target_omega) < min_diff:
                        conjugate_idx = k
                        break
                if conjugate_idx == -1:
                    conjugate_idx = i
                    
            # Set both the mode and its conjugate to zero frequency
            omega_sorted[i] = 0.0
            omega_sorted[conjugate_idx] = 0.0
            
            processed_indices.add(i)
            processed_indices.add(conjugate_idx)
    
    # If specific modes are selected, create new arrays with only those modes
    if selected_mode_indices is not None:
        # Convert selected indices to the sorted space
        selected_indices = sort_indices[selected_mode_indices]
        omega_sorted = omega_sorted[selected_indices]
        Psi_sorted = Psi_sorted[:, selected_indices]
        amplitudes_sorted = amplitudes_sorted[selected_indices]
    
    # Reconstruct using the modified components
    reconstruction = reconstruct_dmd(times, omega_sorted, Psi_sorted, amplitudes_sorted)
    
    # Add mean shape and reshape
    average_shape = average_shape.reshape(1, -1)
    forecast_plus_mean = reconstruction + average_shape
    keypoints = reshape_data(forecast_plus_mean, -1, num_markers, 3)
    
    return keypoints
