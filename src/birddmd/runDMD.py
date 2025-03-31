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
                            eig_constraints = {"imag"}, 
                            min_seq_length: int | None = None, 
                            verbose: bool = True):
    
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
                            d = d )

    

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


# ------------- Run DMD on a Sequence ------------- 

def run_DMD_sequence(seqID:str, 
                     df, 
                     marker_column_names, 
                     average_shape=None, 
                     n_Modes:int=6,
                     min_seq_length= None, 
                     eig_constraints={"imag", "conjugate_pairs"}, 
                     d:int=2):


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
    times, markers = spline_interpolation(times, markers)


    normalised_markers, _ = normalise_data(markers, average_shape)

    dmd_results, forecast = perform_dmd(normalised_markers, times, n_Modes, eig_constraints=eig_constraints, d=d)
    Lambda, Modes, bn, Psi, phase_shifts = reorder_dmd_results(dmd_results, num_markers, n_Modes)

    print(np.round(dmd_results.eigs, 3))

    keypoints = run_forecast(dmd_results, times, average_shape, num_markers)

    markers = markers.reshape(-1, num_markers, 3)

    return times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints



# ----------------------------------------------------
#                   HELPER FUNCTIONS
# ----------------------------------------------------

def spline_interpolation(times, markers):
    """
    Perform spline interpolation to fill missing time points.
    Parameters:
    - times: np.array, array of time points.
    - markers: np.array, array of marker positions corresponding to time points.
    Returns:
    - new_times: np.array, new array of time points with missing points filled.
    - new_markers: np.array, new array of interpolated marker positions.
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

def load_bird_data(bird_name, behaviour, perch_distance=None, bilateral="Bilateral", turn = "Straight"):
    """
    Load bird marker and info data into a DataFrame based on the bird name,
    perch distance, and bilateral status.

    Parameters:
    - bird_name: str, name of the bird.
    - perch_distance: str, distance of the perch, e.g., '12m'. If none, all perch distances are included. 
    - bilateral: bool, True if the condition is bilateral, False otherwise.
    - Turn: str, turn direction, defaults to straight. Only relevant for 9m flights

    Returns:
    - wingbeat_df: DataFrame containing the concatenated info and marker data.
    - marker_column_names: list, loaded list of marker names
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
    Identifies duplicates in a specified column, reports their percentage,
    removes them, and resets the index of the DataFrame.

    Parameters:
    - df: pandas.DataFrame to process.
    - column_name: The name of the column to check for duplicates.

    Returns:
    - df: The modified DataFrame with duplicates removed.
    - ratio_duplicates: The percentage of rows that were duplicates.
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
    Load marker and time data for a given sequence from the source dataframe.
    Creates numpy arrays for DMD.

    Parameters:
    - df: DataFrame containing the concatenated info and marker data.
    - seqID: str, the sequence instance to extract from the dataframe. 
    - marker_column_names: list, loaded marker names that are in the dataframe.
    Returns:
    - markers: np array, marker coordinates from the flight sequence.  
    - time: np array, time data for the flight sequence.
    """

    condition = df['seqID'] == seqID
    markers = df[condition][marker_column_names].to_numpy().astype(np.float64)
    times = df[condition]['time'].to_numpy().astype(np.float64)
    return markers, times

def get_average_shape(marker_column_names):
    
    nMarkers = len(marker_column_names)//3
    hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
    if nMarkers == 8:
        return hawk3d.markers.reshape(24,1).T, nMarkers
    elif nMarkers == 4:
        return hawk3d.right_markers.reshape(12,1).T, nMarkers
    else:
        raise ValueError("Expected 4 markers or 8 (12 or 24 coordinates) but got {nMarkers} markers.")
    

def normalise_data(markers, average_shape):
    """Normalise marker data by subtracting the average shape."""
    return markers - average_shape, average_shape

def perform_dmd(markers, times, n_Modes, eig_constraints={"imag", "conjugate_pairs"}, d=2):
    """Perform Dynamic Mode Decomposition on the marker data."""

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


def run_forecast(dmd_results, times, average_shape, num_markers):
    """Run the forecast to get the predicted markers."""
    reconstruction = reconstruct_dmd(times, 
                                   dmd_results.eigs, 
                                   dmd_results.modes[:num_markers*3,:], 
                                   dmd_results.amplitudes)
    
    forecast_plus_mean = reconstruction + average_shape
    keypoints = forecast_plus_mean.reshape(-1, num_markers, 3)
    
    # Take away the first frame and add an extra frame to the end
    keypoints = np.concatenate([keypoints[1:], keypoints[-1:]], axis=0)
    
    return keypoints


def reorder_dmd_results(dmd_results, num_markers, nModes):
    """Reorder DMD results based on amplitude size and extract components."""
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
    phase_shifts = np.arctan2(-sign_omega*np.imag(Psi), np.real(Psi))
    
    # Get eigenvalues and reorder
    Lambda = dmd_results.eigs[bn]
    
    # Reshape modes into (markers, coordinates, modes) format
    Modes = mode_magnitudes.reshape(num_markers, 3, nModes)

    return np.imag(Lambda), Modes, dmd_results.amplitudes[bn], Psi, phase_shifts

# ------------- Reconstructing Modes ------------- 

def reconstruct_dmd(times, Lambda, Psi, amplitudes):
    """Reconstruct data using phasor notation for DMD."""
    n_modes = len(Lambda)
    reconstruction = np.zeros((len(times), Psi.shape[0]), dtype=complex)
    
    for i in range(0, n_modes, 2):  # Step by 2 for conjugate pairs
        # Calculate magnitude and phase for this mode
        magnitude = np.sqrt(np.real(Psi[:,i])**2 + np.imag(Psi[:,i])**2)
        phase = np.arctan2(-np.imag(Psi[:,i]), np.real(Psi[:,i]))
        
        # Reconstruct using phasor notation
        beta = np.abs(amplitudes[i])
        omega = Lambda[i]
        
        for t_idx, t in enumerate(times):
            reconstruction[t_idx,:] += 2 * beta * magnitude * \
                np.cos(omega * t - phase)
    
    return np.real(reconstruction)


# ------------- Saving DMD results ------------- 

def save_sequence_results(save_path, seqID, times, markers, Lambda, Modes, bn, Psi, keypoints):
    """Save DMD results to a single .npz file for a given sequence ID."""
    np.savez_compressed(f"{save_path}{seqID}_dmd_results.npz", 
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
    """Load DMD results for a specific sequence ID from a .npz file."""
    try:
        data = np.load(os.path.join(save_path, f"{seqID}_dmd_results.npz"), allow_pickle=True)
    except FileNotFoundError:
        print(f"File {save_path}{seqID}_dmd_results.npz not found.")
        return None, None, None, None, None
    
    times = data['times']
    markers = data['markers']
    Lambda = data['Lambda']
    Modes = data['Modes']
    bn = data['bn']
    Psi = data['Psi']
    keypoints = data['forecast_keypoints']
    
    return times, markers, Lambda, Modes, bn, Psi, keypoints


def load_dmd_results(wingbeat_df, save_path):

    # Initialize list to store data dictionaries
    DMD_dict = []

    # For each .npz file in the directory, load the numpy arrays inside and stack them
    for seqID in wingbeat_df['seqID'].unique():
        # Check if the file exists
        if not os.path.exists(f"{save_path}{seqID}_dmd_results.npz"):
            # print(f"File {save_path}{seqID}_dmd_results.npz not found.")
            continue

        times, markers, Lambda, Modes, bn, Psi, keypoints = load_sequence_results(save_path, seqID)

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
            "keypoints": keypoints
        })

        

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(DMD_dict)

