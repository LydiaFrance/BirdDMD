"""
File input/output operations for BirdDMD.

This module handles saving and loading DMD results, including sequence data,
marker data, and DMD analysis results. It provides functions for persisting
analysis results to disk and loading them back for further analysis.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any

# Constants
SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'samples'))

# Error handling
class IOError(Exception):
    """Base class for IO-related errors."""
    pass

class FileNotFoundError(IOError):
    """Raised when a required file is not found."""
    pass

class DataValidationError(IOError):
    """Raised when data validation fails during IO operations."""
    pass

def save_sequence_results(save_path: str, 
                         seqID: str, 
                         times: np.ndarray, 
                         markers: np.ndarray, 
                         Lambda: np.ndarray, 
                         Modes: np.ndarray, 
                         bn: np.ndarray, 
                         Psi: np.ndarray, 
                         keypoints: np.ndarray) -> None:
    """
    Save DMD results and metadata for a single sequence to a compressed .npz file.

    Args:
        save_path: Directory to save the file
        seqID: Sequence ID, used in the filename
        times: Time vector
        markers: Original marker data (reshaped)
        Lambda: Imaginary eigenvalues (frequencies)
        Modes: Mode magnitudes
        bn: Amplitudes
        Psi: Complex modes
        keypoints: Reconstructed full trajectory

    Raises:
        IOError: If there are issues creating or writing to the file
    """
    filename = os.path.join(save_path, f"{seqID}_dmd_results.npz")

    try:
        np.savez_compressed(filename, 
                          times=times,
                          markers=markers,
                          Lambda=Lambda, 
                          Modes=Modes, 
                          bn=bn, 
                          Psi=Psi, 
                          forecast_keypoints=keypoints, 
                          allow_pickle=True)
    except Exception as e:
        raise IOError(f"Failed to save sequence results to {filename}: {str(e)}")

def load_sequence_results(save_path: str, seqID: str) -> Tuple[Optional[np.ndarray], ...]:
    """
    Load DMD results for a specific sequence ID from a .npz file.

    Args:
        save_path: Directory where the .npz file is located
        seqID: Sequence ID to load

    Returns:
        Tuple containing loaded data: (times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints)
        Returns None for all if the file is not found or phase_shifts key is missing

    Raises:
        FileNotFoundError: If the file doesn't exist
        DataValidationError: If the file exists but is corrupted or missing essential data
    """
    filename = os.path.join(save_path, f"{seqID}_dmd_results.npz")
    
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found.")
    except Exception as e:
        raise DataValidationError(f"Error loading {filename}: {str(e)}")
    
    # Extract data with validation
    times = data.get('times')
    markers = data.get('markers')
    Lambda = data.get('Lambda')
    Modes = data.get('Modes')
    bn = data.get('bn')
    Psi = data.get('Psi')
    phase_shifts = data.get('phase_shifts')
    keypoints = data.get('forecast_keypoints')

    if times is None or Lambda is None:
        raise DataValidationError(f"File {filename} is missing essential data.")
    
    return times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints

def load_dmd_results(wingbeat_df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Loads DMD results for all sequences present in a DataFrame from a specified directory.

    Args:
        wingbeat_df: DataFrame containing seqIDs to load
        save_path: Directory containing the saved _dmd_results.npz files

    Returns:
        DataFrame where each row corresponds to a sequence and columns contain the loaded DMD results

    Raises:
        DataValidationError: If there are issues with the data format or missing files
    """
    # Initialize list to store data dictionaries
    DMD_dict = []

    # For each .npz file in the directory, load the numpy arrays inside and stack them
    for seqID in wingbeat_df['seqID'].unique():
        try:
            times, markers, Lambda, Modes, bn, Psi, phase_shifts, keypoints = load_sequence_results(save_path, seqID)
            
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
        except (FileNotFoundError, DataValidationError) as e:
            print(f"Warning: {str(e)}")
            continue

    if not DMD_dict:
        raise DataValidationError("No valid DMD results found in the specified directory.")

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(DMD_dict)
