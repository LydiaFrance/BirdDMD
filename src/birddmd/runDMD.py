import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing

import warnings
warnings.filterwarnings("ignore")

from morphing_birds import Hawk3D

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'samples'))


def run_single_wingbeat_dmd(bird_name: str, 
                            perch_dist: str, 
                            turn: str = "Straight", 
                            behaviour: str = "Initial",
                            bilateral: str = "Bilateral", 
                            seqID: str | None = None,
                            n_modes: int = 6,
                            d: int = 2,
                            eig_constraints = {"imag", "conjugate_pairs"}, 
                            min_seq_length: int | None = None, 
                            interpolate: bool = False,
                            verbose: bool = True):
    """
    Runs DMD analysis on a single, specific wingbeat sequence identified by various parameters.

    Loads data, gets average shape, and calls run_DMD_sequence.

    Parameters:
        bird_name: Name of the hawk in the dataset.
        perch_dist: Perch distance condition (e.g., '9m').
        turn: Turn condition ('Straight', 'Left', 'Right'), relevant for some datasets.
        behaviour: Flight behaviour description (e.g., 'Initial').
        bilateral: Data type ('Bilateral' or 'Unilateral').
        seqID: Specific sequence ID to process. If None, the first sequence in the filtered data is used.
        n_modes: Number of DMD modes to compute.
        d: Hankel matrix delay parameter.
        eig_constraints: Constraints for BOPDMD eigenvalues (e.g., {"imag", "conjugate_pairs"}).
        min_seq_length: Minimum number of frames required to process a sequence.
        verbose: If True, print status messages.

    Returns:
        Output from run_DMD_sequence for the specified sequence.
    """
    df, marker_column_names = load_bird_data(
        bird_name = bird_name, 
        behaviour = behaviour, 
        perch_distance = perch_dist,
        bilateral = bilateral,
        turn = turn
    )

    if seqID is None:
        """Pick first sequence if not specified."""
        seqID = df['seqID'].iloc[0]
    
    if verbose:
        print(f"Running DMD for {bird_name}, {perch_dist}m, {turn} turn, seqID: {seqID}")
    
    average_shape, _ = get_average_shape(marker_column_names)

    return run_DMD_sequence(seqID = seqID,
                            df = df, 
                            marker_column_names = marker_column_names,
                            average_shape = average_shape,
                            n_Modes = n_modes, 
                            min_seq_length = min_seq_length, 
                            eig_constraints = eig_constraints,
                            d = d, 
                            interpolate = interpolate)

    

# ------------- Loop DMD every Sequence ------------- 

def dmd_loop_seqs(bird_name,  
                  save_path,
                  behaviour,
                  perch_distance=None, 
                  bilateral="Bilateral", 
                  turn = "Straight", 
                  n_Modes=6, 
                  min_seq_length= None,
                  eig_constraints={"imag", "conjugate_pairs"}, 
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


    average_shape, _ = get_average_shape(marker_column_names)

    df = remove_time_duplicates(df)

    for seqID in df['seqID'].unique():

        times, markers, Lambda, Modes, bn, Psi, keypoints = run_DMD_sequence(seqID, df, marker_column_names, average_shape, n_Modes, min_seq_length, eig_constraints, d)
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
                     eig_constraints={"imag", "conjugate_pairs"}, 
                     d:int=2, 
                     interpolate: bool = False):

    """
    Runs the complete DMD analysis pipeline for a single sequence.

    Includes data loading, optional interpolation, normalization, DMD computation,
    results reordering, and forecasting the full reconstruction.

    Parameters:
        seqID: The ID of the sequence to process.
        df: DataFrame containing the full dataset.
        marker_column_names: List of column names corresponding to marker coordinates.
        average_shape: The mean shape to subtract/add back. If None, it's calculated.
        n_Modes: Number of DMD modes.
        min_seq_length: Minimum sequence length required.
        eig_constraints: Constraints for BOPDMD eigenvalues.
        d: Hankel matrix delay parameter.

    Returns:
        tuple: Contains times, original_markers_reshaped, Lambda (imaginary eigenvalues),
               Modes (magnitudes), bn (amplitudes), Psi (complex modes), phase_shifts,
               keypoints (reconstructed trajectory). Returns None for all if sequence is too short.
    """

    markers, times = load_sequence_data(df, seqID, marker_column_names)

    if average_shape is None:
        average_shape, num_markers = get_average_shape(marker_column_names)
    else:
        num_markers = average_shape.shape[1]//3


    if min_seq_length == None:
        min_seq_length = n_Modes+1

    if markers.shape[0] <= min_seq_length:
        print(f"Skipping sequence {seqID} due to insufficient timesteps.")
        return None, None, None, None, None, None, None


    # Perform spline interpolation to fill in missing time points
    if interpolate:
        times, markers = spline_interpolation(times, markers)


    normalised_markers, _ = normalise_data(markers, average_shape)

    dmd_results, _ = perform_dmd(normalised_markers, times, n_Modes, eig_constraints=eig_constraints, d=d)
    
    # Need to handle case where dmd_results might be None if fit fails, though unlikely here
    if dmd_results is None:
         print(f"DMD fit failed for sequence {seqID}.")
         num_return_values = 8
         return (None,) * num_return_values

    
    Lambda, Modes, bn, Psi, phase_shifts = reorder_dmd_results(dmd_results, num_markers, n_Modes)

    
    print(f"Complex Eigenvalues (log) for {seqID}: {np.round(dmd_results.eigs, 3)}") # Changed print statement slightly

    keypoints = run_forecast(dmd_results, times, average_shape, num_markers)

    markers = markers.reshape(-1, num_markers, 3)

    return times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints



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

def get_average_shape(marker_column_names):
    """
    Loads the mean hawk shape from a standard file based on the number of markers.

    Parameters:
        marker_column_names: List of marker columns to infer the number of markers (4 or 8).

    Returns:
        average_shape: The mean shape as a (1, n_coords) NumPy array.
        nMarkers: The number of markers (4 or 8).
    """
    nMarkers = len(marker_column_names)//3
    hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
    if nMarkers == 8:
        return hawk3d.markers.reshape(24,1).T, nMarkers
    elif nMarkers == 4:
        return hawk3d.right_markers.reshape(12,1).T, nMarkers
    else:
        raise ValueError("Expected 4 markers or 8 (12 or 24 coordinates) but got {nMarkers} markers.")
    

def normalise_data(markers, average_shape):
    """
    Center the marker data by subtracting the average shape.

    Parameters:
        markers: Marker data array (n_frames, n_coords).
        average_shape: The mean shape (1, n_coords).

    Returns:
        normalized_markers: Centered marker data.
        average_shape: The average shape passed in (for convenience).
    """
    return markers - average_shape, average_shape


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


def perform_dmd(markers, times, n_Modes, eig_constraints={"imag", "conjugate_pairs"}, d=2):
    """
    Performs Bagging Online Proper Orthogonal Decomposition DMD (BOPDMD) with Hankel preprocessing.

    Parameters:
        markers: Normalised marker data, shape (n_frames, n_coords).
        times: Timestamps corresponding to markers.
        n_Modes: Target rank (number of modes) for DMD.
        eig_constraints: Constraints applied to eigenvalues during optimization.
        d: Delay embedding parameter for Hankel matrix.

    Returns:
        dmd_results: Fitted pydmd BOPDMD object.
        forecast: Reconstructed trajectory using dmd_results.forecast (Note: run_forecast uses reconstruct_dmd).
                 This return value might be redundant if reconstruct_dmd is always used downstream.
    """
    markers = markers.T  # Transpose data
    dmd_results = hankel_preprocessing(BOPDMD(svd_rank=n_Modes, 
                                                eig_constraints=eig_constraints), 
                                                d=d)
    
    dmd_results.fit(markers, t=times[1:])
    forecast = dmd_results.forecast(times)
    return dmd_results, forecast


# 2024-12-12 Correcting based on conversation with Karl
# def run_forecast(forecast, average_shape, num_markers):
#     """Run the forecast to get the predicted markers."""
#     forecast = np.real(forecast[:num_markers*3,:])
#     forecast_plus_mean = forecast.T + average_shape

#     keypoints = forecast_plus_mean.reshape(-1, num_markers, 3)

#     # Take away the first frame and add an extra frame to the end
#     keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)

#     return keypoints

# 2024-12-12 Correcting based on conversation with Karl

# def reorder_dmd_results(dmd_results, num_markers, nModes):
#     """Reorder DMD results based on amplitude size and extract components."""
#     bn = np.argsort(dmd_results.amplitudes)[::-1]
#     Psi = dmd_results.modes
#     Psi = Psi[:num_markers*3, bn]
    
#     Lambda = np.imag(dmd_results.eigs)[bn]
#     Modes = np.real(Psi).reshape(num_markers, 3, nModes)

#     return Lambda, Modes, dmd_results.amplitudes[bn], Psi

# ====================================================
#       Helper Functions: DMD Results Processing
# ====================================================
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
    # Pass the imaginary part of eigenvalues as omega (frequencies)
    reconstruction = reconstruct_dmd(times,
                                   np.imag(dmd_results.eigs), # Pass frequencies
                                   dmd_results.modes[:n_coords,:],
                                   dmd_results.amplitudes)

    # Add mean shape and reshape
    # Ensure reconstruction shape is (n_times, n_coords) before adding mean shape
    if reconstruction.shape[1] != average_shape.shape[1]:
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


def reorder_dmd_results(dmd_results, num_markers, nModes):
    """
    Reorders DMD results by amplitude magnitude and extracts useful components.

    Calculates mode magnitudes and phase shifts using phasor interpretation.

    Parameters:
        dmd_results: Fitted pydmd object.
        num_markers: Number of markers (to correctly slice modes).
        nModes: Number of modes computed.

    Returns:
        Lambda_imag: Imaginary part of eigenvalues (frequencies), sorted by amplitude.
        Modes_mag: Magnitudes of modes, sorted, shape (num_markers, 3, nModes).
        bn: Amplitudes, sorted by magnitude.
        Psi: Complex modes, sorted by amplitude, shape (n_coords, nModes).
        phase_shifts: Phase shifts of modes in radians, sorted, shape (n_coords, nModes).
    """
    # Sort by amplitude magnitude
    bn = np.argsort(np.abs(dmd_results.amplitudes))[::-1]
    
    # Get modes and reorder
    Psi = dmd_results.modes
    Psi = Psi[:num_markers*3, bn]

    # Fixes the sign of the phase shift
    omega = np.imag(dmd_results.eigs)
    sign_omega = np.sign(omega)
    
    # Calculate mode magnitudes (phasor notation)
    mode_magnitudes = np.sqrt(np.real(Psi)**2 + np.imag(Psi)**2)
    
    # Calculate phase shifts
    # Thanks Karl!
    # Calculate phase shifts (atan2 handles quadrants correctly)
    # Phase = atan2(-imag_part * sign(frequency), real_part) ensures phase leads for positive freq
    phase_shifts = np.arctan2(-sign_omega * np.imag(Psi), np.real(Psi)) # Added sign_omega factor

    # Get eigenvalues and reorder
    Lambda = dmd_results.eigs[bn]
    
    # Reshape modes into (markers, coordinates, modes) format
    Modes = mode_magnitudes.reshape(num_markers, 3, nModes)

    return np.imag(Lambda), Modes, dmd_results.amplitudes[bn], Psi, phase_shifts

# ------------- Reconstructing Modes ------------- 


def reconstruct_dmd(times, omega, Psi, amplitudes):
    """
    Reconstructs data using phasor notation for DMD modes (assumes conjugate pairs).

    Sums the contribution of each mode pair (A * cos(omega*t - phase)).

    Parameters:
        times: Time vector for reconstruction.
        omega: Imaginary part of eigenvalues (frequencies). Assumes sorted conjugate pairs.
        Psi: Complex DMD modes. Assumes sorted corresponding to omega/amplitudes.
        amplitudes: Complex DMD amplitudes. Assumes sorted.

    Returns:
        reconstruction: Real-valued reconstructed data (n_times, n_coords).
    """
    n_coords = Psi.shape[0]
    n_modes = len(omega)
    reconstruction = np.zeros((len(times), n_coords), dtype=float) # Output is real

    # Sort by amplitude magnitude to pair modes correctly if not already done
    # Note: reorder_dmd_results should already provide sorted inputs
    sort_indices = np.argsort(np.abs(amplitudes))[::-1]
    omega_sorted = omega[sort_indices]
    Psi_sorted = Psi[:, sort_indices]
    amplitudes_sorted = amplitudes[sort_indices]
    
    processed_indices = set()

    for i in range(n_modes):
        if i in processed_indices:
            continue

        # Find conjugate pair (or itself if real eigenvalue/zero frequency)
        # Look for opposite frequency, allow small tolerance
        target_omega = -omega_sorted[i]
        conjugate_idx = -1
        min_diff = 1e-6 # Tolerance for finding the conjugate pair frequency
        
        # Efficiently find potential conjugate index (if modes are truly paired)
        if i % 2 == 0 and i + 1 < n_modes:
             if np.abs(omega_sorted[i+1] - target_omega) < min_diff:
                 conjugate_idx = i + 1
        # Fallback search if not found immediately (covers edge cases)
        if conjugate_idx == -1:
            for k in range(i + 1, n_modes):
                if k not in processed_indices and np.abs(omega_sorted[k] - target_omega) < min_diff:
                    conjugate_idx = k
                    break
            # If still no conjugate found, treat as a non-paired mode (e.g., zero frequency)
            if conjugate_idx == -1:
                 conjugate_idx = i # Pair with itself

        # Calculate magnitude and phase for the primary mode `i`
        # Use the sign of the frequency consistent with reorder_dmd_results phase calc
        sign_omega = np.sign(omega_sorted[i]) if np.abs(omega_sorted[i]) > 1e-9 else 1
        magnitude = np.abs(Psi_sorted[:, i]) # Magnitude is sqrt(real^2 + imag^2)
        phase = np.arctan2(-sign_omega * np.imag(Psi_sorted[:, i]), np.real(Psi_sorted[:, i]))
        
        # Amplitude (use magnitude of the complex amplitude)
        beta = np.abs(amplitudes_sorted[i])
        freq = np.abs(omega_sorted[i]) # Use absolute frequency

        # Add contribution: 2 * beta * magnitude * cos(omega*t - phase)
        # If it's a zero-frequency mode paired with itself (i == conjugate_idx), avoid double counting
        factor = 1.0 if i == conjugate_idx else 2.0

        for t_idx, t in enumerate(times):
             reconstruction[t_idx, :] += factor * beta * magnitude * np.cos(freq * t - phase)

        # Mark both modes as processed
        processed_indices.add(i)
        processed_indices.add(conjugate_idx) # Safe even if i == conjugate_idx

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


def reconstruct_specific_modes(times, eigs_sorted, Psi_sorted, amplitudes_sorted, reordered_mode_indices):
    """
    Reconstructs the trajectory using only a specified subset of DMD modes,
    based on their rank after sorting by amplitude magnitude.

    Uses the reordered DMD components corresponding to the provided indices.

    Parameters:
        times: Time vector for reconstruction.
        eigs_sorted: Complex eigenvalues, sorted by amplitude magnitude
                     (output from reorder_dmd_results).
        Psi_sorted: Complex mode vectors, sorted by amplitude magnitude
                    (output from reorder_dmd_results).
        amplitudes_sorted: Complex amplitudes, sorted by amplitude magnitude
                           (output from reorder_dmd_results).
        reordered_mode_indices: List or array of integer indices (0-based) specifying
                                which modes to use *after* they have been sorted by
                                amplitude. E.g., [0, 1] uses the two modes with the
                                highest amplitudes.

    Returns:
        reconstruction: Complex-valued reconstructed data from selected modes (n_coords, n_times).
                        Take np.real() for the physical trajectory.
    """
    n_coords = Psi_sorted.shape[0]
    reconstruction = np.zeros((n_coords, len(times)), dtype=complex)

    # Check if inputs are consistent
    if not (len(eigs_sorted) == Psi_sorted.shape[1] == len(amplitudes_sorted)):
        raise ValueError("Input arrays (eigs, Psi, amplitudes) must have consistent number of modes.")

    max_index = max(reordered_mode_indices) if reordered_mode_indices else -1
    if max_index >= len(eigs_sorted):
         raise IndexError(f"Mode index {max_index} out of bounds for {len(eigs_sorted)} available sorted modes.")

    for j in reordered_mode_indices:
        # Use the j-th component from the *sorted* arrays
        mode = Psi_sorted[:, j:j+1]       # Shape (n_coords, 1)
        eig = eigs_sorted[j]              # Scalar
        amplitude = amplitudes_sorted[j]  # Scalar

        # Time evolution vector for this mode
        # Use the complex eigenvalue directly: exp(eig*t) = exp(sigma*t) * exp(j*omega*t)
        # np.outer creates shape (1, n_times) correctly
        time_dynamics = np.exp(np.outer(eig, times))

        # Add this mode's contribution: amplitude * mode_vector * time_dynamics_vector
        reconstruction += amplitude * (mode @ time_dynamics) # Matrix multiplication handles shapes

    # Transpose the result to be (n_times, n_coords) consistent with reconstruct_dmd
    return reconstruction.T


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
    left_right_bool = np.concatenate([
        np.zeros(n_frames, dtype=bool),  # False (0) for left
        np.ones(n_frames, dtype=bool)    # True (1) for right
    ])

    return dmd_reconstruction_flat, left_right_bool


def project_into_pca_space(new_data, mu, principal_components):
    """
    Projects new shape data (e.g., from DMD reconstruction) into an existing PCA space.

    Assumes the new data is already flattened and centered appropriately if needed,
    or that `mu` is the correct mean to subtract.

    Parameters:
        new_data: New data to project, shape (n_samples, n_features),
                  where n_features matches the PCA space (e.g., n_unilateral_markers * 3).
        mu: Mean shape used for the original PCA, shape (1, n_features) or (n_features,).
        principal_components: PCA components (eigenvectors), shape (n_components, n_features).

    Returns:
        scores_new: Projected data (scores) in the PCA space, shape (n_samples, n_components).
    """
    # Flatten the mean shape if it's not already flat
    mu_flat = mu.flatten() # shape: (n_features,)

    # Center the new data by subtracting the mean
    new_data_centered = new_data - mu_flat

    # Project onto principal components: scores = centered_data @ components.T
    scores_new = np.dot(new_data_centered, principal_components.T)

    return scores_new


def create_scores_info_df(scores, seq_info_df, left_right_bool=None):
    """
    Combines PCA scores with sequence metadata into a DataFrame.

    Handles potential mismatch in frame counts if data was duplicated/split (e.g., unilateral).

    Parameters:
        scores: PCA scores array, shape (n_frames_scores, n_components).
        seq_info_df: DataFrame containing metadata for the original sequence (n_frames_original, n_info_cols).
        left_right_bool: Optional boolean array (length n_frames_scores) indicating left/right side (True=Right).
                         Used to correctly duplicate/assign metadata.

    Returns:
        scores_df: DataFrame combining scores and metadata.
    """
    num_components = scores.shape[1]
    PC_names = [f'PC{i:02}' for i in range(1, num_components + 1)] # Use range for 0-based index -> PC01

    # Create a DataFrame from scores
    scores_df = pd.DataFrame(scores, columns=PC_names)

    # Handle metadata alignment
    n_frames_scores = scores.shape[0]
    n_frames_original = seq_info_df.shape[0]

    # If scores have twice the frames (likely unilateral split), duplicate metadata
    if n_frames_scores == 2 * n_frames_original and left_right_bool is not None:
        meta_df_aligned = pd.concat([seq_info_df.reset_index(drop=True),
                                     seq_info_df.reset_index(drop=True)],
                                    ignore_index=True)
        # Add 'Left' column based on the boolean flag (0 for Left, 1 for Right)
        meta_df_aligned['Left'] = left_right_bool.astype(int)

    # If frame counts match directly
    elif n_frames_scores == n_frames_original:
        meta_df_aligned = seq_info_df.reset_index(drop=True)
        # If left_right_bool is provided unexpectedly, or if needed for single-sided data
        if left_right_bool is not None:
             meta_df_aligned['Left'] = left_right_bool.astype(int)

    else:
        raise ValueError(f"Score frames ({n_frames_scores}) mismatch original seq frames ({n_frames_original}).")

    # Concatenate scores and aligned metadata
    combined_df = pd.concat([meta_df_aligned, scores_df], axis=1)

    return combined_df