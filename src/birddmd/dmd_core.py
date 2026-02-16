"""
Core DMD computation and analysis functions for BirdDMD.

This module contains the core Dynamic Mode Decomposition (DMD) functionality,
including the main DMD computation, sequence processing, and various specialized
DMD analysis functions for different types of input data.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing

from .data_handling import (
    get_average_shape,
    load_bird_data,
    load_sequence_data,
    normalise_data,
    remove_time_duplicates,
    spline_interpolation,
    validate_average_shape,
    validate_marker_data,
    validate_times,
)
from .dmd_reconstruction import reconstruct_dmd
from .exceptions import ComputationError, ValidationError
from .io import save_sequence_results

warnings.filterwarnings("ignore")

# Constants
DEFAULT_N_MODES = 10
DEFAULT_D = 2
DEFAULT_EIG_CONSTRAINTS = {"conjugate_pairs"}
DEFAULT_MIN_SEQ_LENGTH = None
DIMENSION_2D = 2
DIMENSION_3D = 3
MARKER_RESHAPE_THRESHOLD = 8


def _compute_dmd(
    data: np.ndarray,
    times: np.ndarray,
    n_modes: int,
    eig_constraints: set[str],
    d: int,
    debug: bool = False,
) -> tuple[BOPDMD | None, np.ndarray]:
    """Compute DMD on preprocessed data.

    Args:
        data: Input data array in shape (n_variables, n_timesteps)
        times: Array of timestamps
        n_modes: Number of DMD modes to compute
        eig_constraints: Constraints for BOPDMD eigenvalues
        d: Hankel matrix delay parameter
        debug: Whether to print debug information
    Returns:
        Tuple of (dmd_results, processed_data)
        dmd_results is None if computation fails

    Raises:
        ComputationError: If DMD computation fails
    """
    try:
        if debug:
            # Debug prints
            print("\nDMD Execution Debug:")
            print(f"Data shape: {data.shape}")
            print(f"Times shape: {times.shape}")
            print(f"Number of modes: {n_modes}")
            print(f"Eigenvalue constraints: {eig_constraints}")
            print(f"Delay parameter d: {d}")
            print(f"Data min/max: {np.min(data):.3f}/{np.max(data):.3f}")
            print(f"Times min/max: {np.min(times):.3f}/{np.max(times):.3f}")

        # Ensure data is in correct shape (n_variables, n_timesteps)
        if data.shape[0] > data.shape[1]:
            data = data.T

        # Initialise and fit BOPDMD
        dmd = hankel_preprocessing(
            BOPDMD(svd_rank=n_modes, eig_constraints=eig_constraints), d=d
        )

        # Ensure times match processed data shape
        if len(times) != data.shape[1]:
            times = times[: data.shape[1]]

        if debug:
            print(f"Time size: {times.shape}")
            print(f"Data shape: {data.shape}")

        # Check time vector is strictly increasing and unique
        dt = np.mean(np.diff(times))
        if np.any(dt <= 0):
            msg = "Time vector must be strictly increasing and unique"
            raise ValidationError(msg)

        if debug:
            print("\nFitting DMD...")

        dmd.fit(data, t=times[1:])  # Use times[1:] for fitting

        if debug:
            print("DMD fit complete")
            print(f"Number of eigenvalues: {len(dmd.eigs)}")
            print(f"Eigenvalues: {np.round(dmd.eigs, 3)}")

        return dmd, data

    except Exception as e:
        msg = f"DMD computation failed: {e!s}"
        raise ComputationError(msg) from e


def reorder_dmd_results(
    dmd_results: BOPDMD,
    num_markers: int,
    order_by: str = "amplitude",
    reshape_modes: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reorder DMD results by amplitude or frequency.

    Args:
        dmd_results: Fitted BOPDMD object
        num_markers: Number of markers or variables
        n_modes: Number of modes computed
        order_by: Ordering method ('amplitude' or 'frequency')
        reshape_modes: Whether to reshape modes for marker data

    Returns:
        Tuple of (Lambda, Modes, bn, Psi, phase_shifts, sort_idx)

    Raises:
        ValueError: If order_by is invalid
    """
    if order_by not in ["amplitude", "frequency"]:
        msg = f"order_by must be either 'amplitude' or 'frequency', got '{order_by}'"
        raise ValueError(msg)

    # Get frequencies and determine sort order
    frequencies = np.imag(dmd_results.eigs)
    sign_omega = np.sign(frequencies)

    if order_by == "amplitude":
        sort_idx = np.argsort(np.abs(dmd_results.amplitudes))[::-1]
    else:  # frequency
        sort_idx = np.argsort(np.abs(frequencies))[::-1]

    # Get and reorder modes
    Psi = dmd_results.modes
    n_vars = num_markers if not reshape_modes else num_markers * DIMENSION_3D
    Psi = Psi[:n_vars, sort_idx]

    # Get actual number of modes from the data
    actual_n_modes = Psi.shape[1]

    mode_magnitudes = np.sqrt(np.real(Psi) ** 2 + np.imag(Psi) ** 2)

    # Handle marker data vs time series data
    if reshape_modes and num_markers <= MARKER_RESHAPE_THRESHOLD:  # Marker data case
        Modes = mode_magnitudes.reshape(num_markers, DIMENSION_3D, actual_n_modes)
    else:  # Time series case
        Modes = mode_magnitudes

    # Reorder other components
    Lambda = frequencies[sort_idx]
    bn = dmd_results.amplitudes[sort_idx]
    phase_shifts = np.arctan2(-sign_omega[sort_idx] * np.imag(Psi), np.real(Psi))

    return Lambda, Modes, bn, Psi, phase_shifts, sort_idx


def run_dmd(
    data: np.ndarray,
    times: np.ndarray | None = None,
    n_modes: int = DEFAULT_N_MODES,
    d: int = DEFAULT_D,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    average_shape: np.ndarray | None = None,
    num_markers: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, np.ndarray]:
    """Main public interface for running DMD analysis on time series data.

    This function handles both marker data (3D coordinates) and generic time
        series data
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

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
    """
    # Check if this is time series data (n_variables, n_timesteps)
    is_timeseries = len(data.shape) == DIMENSION_2D and data.shape[0] < data.shape[1]

    if is_timeseries:
        # For time series data, transpose to (n_variables, n_timesteps)
        data_flat = data.T if data.shape[0] > data.shape[1] else data
        n_frames = data_flat.shape[1]
        n_coords = data_flat.shape[0]
    else:
        # For marker data, validate and transform
        data_flat, n_frames, _, n_coords = validate_marker_data(data)

    # Validate times
    times = validate_times(times, n_frames)

    # For marker data, validate average shape
    if not is_timeseries:
        if num_markers is None:
            msg = "num_markers must be provided for marker data"
            raise ValidationError(msg)
        average_shape = validate_average_shape(
            average_shape, n_coords, is_timeseries=False
        )

    if verbose:
        print(f"Running DMD with {n_modes} modes, delay d={d}")
        print(f"Input shape: {data_flat.shape}")

    # Normalise data if it's marker data
    if not is_timeseries:
        data_flat, _ = normalise_data(data_flat)

    try:
        # Perform DMD computation
        dmd_results, _ = _compute_dmd(
            data_flat, times, n_modes, eig_constraints=eig_constraints, d=d
        )

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
            reshape_modes=(num_markers is not None),
        )

        if verbose:
            print(f"Complex Eigenvalues (log): {np.round(dmd_results.eigs, 3)}")
            print(f"Number of modes in Psi: {Psi.shape[1]}")

        # Reconstruct time series
        reconstruction = reconstruct_dmd(
            times,
            np.imag(dmd_results.eigs),
            dmd_results.modes[:n_coords, :],
            dmd_results.amplitudes,
        )

        # If this is marker data, reshape the reconstruction
        if not is_timeseries:
            reconstruction = reconstruction.reshape(-1, num_markers, 3)
            if average_shape is not None:
                reconstruction = reconstruction + average_shape.reshape(
                    -1, num_markers, 3
                )

        return Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction

    except Exception as e:
        msg = f"DMD analysis failed: {e!s}"
        raise ComputationError(msg) from e


def run_sequence_dmd(
    bird_name: str,
    perch_dist: str,
    turn: str,
    behaviour: str,
    seqID: str,
    n_modes: int = DEFAULT_N_MODES,
    d: int = DEFAULT_D,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    min_seq_length: int | None = DEFAULT_MIN_SEQ_LENGTH,
    interpolate: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray | None, ...]:
    """Run DMD analysis on a specific sequence from the dataset.

    This function handles loading the data and running DMD analysis.

    Args:
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
        Tuple of (times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results,
                  keypoints)
        Returns None for all if sequence is too short or DMD fails

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
    """
    try:
        # Load data

        df, marker_column_names = load_bird_data(
            bird_name=bird_name,
            behaviour=behaviour,
            perch_distance=perch_dist,
            turn=turn,
        )

        if verbose:
            print(
                f"Running DMD for {bird_name}, {perch_dist}m,"
                f"{turn} turn, seqID: {seqID}"
            )

        # Get average shape
        n_markers = len(marker_column_names) // 3
        average_shape = get_average_shape(n_markers)

        # Load sequence data

        markers, times = load_sequence_data(df, seqID, marker_column_names)

        if min_seq_length is None:
            min_seq_length = n_modes + 1

        if markers.shape[0] <= min_seq_length:
            if verbose:
                print(f"Sequence {seqID} too short ({markers.shape[0]} frames)")
            return None, None, None, None, None, None, None, None, None

        # Perform spline interpolation if requested
        if interpolate:
            times, markers = spline_interpolation(times, markers)

        # Reshape markers to (n_frames, n_markers, 3)
        markers = markers.reshape(-1, n_markers, 3)

        # Run DMD analysis
        Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_dmd(
            data=markers,
            times=times,
            n_modes=n_modes,
            d=d,
            eig_constraints=eig_constraints,
            average_shape=average_shape,
            num_markers=n_markers,
            verbose=verbose,
        )

        if dmd_results is None:
            return None, None, None, None, None, None, None, None, None

        return (
            times,
            markers,
            Lambda,
            Modes,
            bn,
            Psi,
            phase_shifts,
            dmd_results,
            keypoints,
        )

    except Exception as e:
        msg = f"Sequence DMD analysis failed: {e!s}"
        raise ComputationError(msg) from e


def run_marker_dmd(
    markers: np.ndarray,
    times: np.ndarray | None = None,
    average_shape: np.ndarray | None = None,
    n_modes: int = DEFAULT_N_MODES,
    d: int = DEFAULT_D,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    verbose: bool = True,
) -> tuple[np.ndarray, ...]:
    """Run DMD analysis directly on marker data.

    This function assumes the data is already loaded and preprocessed.

    Args:
        markers: Marker data array, shape (n_frames, n_markers, 3)
        times: Optional array of timestamps
        average_shape: Mean shape to subtract/add back
        n_modes: Number of DMD modes
        d: Hankel matrix delay parameter
        eig_constraints: Constraints for BOPDMD eigenvalues
        verbose: Whether to print status messages

    Returns:
        Tuple of (Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction)

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
    """
    if len(markers.shape) != DIMENSION_3D:
        msg = (
            f"Markers must be 3D array (n_frames, n_markers, 3),"
            f"got shape {markers.shape}"
        )
        raise ValidationError(msg)

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
        verbose=verbose,
    )


def run_timeseries_dmd(
    data: np.ndarray,
    times: np.ndarray | None = None,
    n_modes: int = DEFAULT_N_MODES,
    d: int = DEFAULT_D,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    verbose: bool = True,
) -> tuple[np.ndarray, ...]:
    """
    Run DMD analysis on generic time series data.
    This function handles any 2D time series data (e.g., PCA scores).

    Args:
        data: Time series data array, shape (n_timesteps, n_variables)
        times: Optional array of timestamps
        n_modes: Number of DMD modes
        d: Hankel matrix delay parameter
        eig_constraints: Constraints for BOPDMD eigenvalues
        verbose: Whether to print status messages

    Returns:
        Tuple of (Lambda, Modes, bn, Psi, phase_shifts, dmd_results, reconstruction)

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
    """
    if len(data.shape) != DIMENSION_2D:
        msg = (
            f"Data must be 2D array (n_timesteps, n_variables), got shape {data.shape}"
        )
        raise ValidationError(msg)

    return run_dmd(
        data=data,
        times=times,
        n_modes=n_modes,
        d=d,
        eig_constraints=eig_constraints,
        verbose=verbose,
    )


def run_single_wingbeat_dmd(
    bird_name: str,
    perch_dist: str,
    turn: str = "Straight",
    behaviour: str = "Initial",
    seqID: str | None = None,
    n_modes: int = DEFAULT_N_MODES,
    d: int = DEFAULT_D,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    min_seq_length: int | None = DEFAULT_MIN_SEQ_LENGTH,
    interpolate: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray | None, ...]:
    """
    Legacy function that maintains backward compatibility.
    Now uses run_sequence_dmd internally.

    Args:
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
        Tuple of (times, markers, Lambda, Modes, bn, Psi, phase_shifts,
                  dmd_results, keypoints)
        Returns None for all if sequence is too short or DMD fails

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
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
        verbose=verbose,
    )


def dmd_loop_seqs(
    bird_name: str,
    save_path: str,
    behaviour: str,
    perch_distance: str | None = None,
    bilateral: str = "Bilateral",
    turn: str = "Straight",
    n_Modes: int = DEFAULT_N_MODES,
    min_seq_length: int | None = DEFAULT_MIN_SEQ_LENGTH,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    d: int = DEFAULT_D,
) -> None:
    """
    Loops through all unique sequences in a dataset, runs DMD, and saves results.

    Args:
        bird_name: Name of the bird dataset
        save_path: Directory path to save the DMD results (.npz files)
        behaviour: Flight behaviour description
        perch_distance: Perch distance condition. If None, uses all distances
        bilateral: Data type ('Bilateral' or 'Unilateral')
        turn: Turn condition
        n_Modes: Number of DMD modes
        min_seq_length: Minimum sequence length to process
        eig_constraints: Constraints for BOPDMD eigenvalues
        d: Hankel matrix delay parameter

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
    """
    try:
        df, marker_column_names = load_bird_data(
            bird_name, behaviour, perch_distance, bilateral, turn
        )

        seqID_counts = df["seqID"].value_counts()

        # Create a histogram
        plt.figure(figsize=(3, 3))
        plt.hist(seqID_counts, bins=20)
        plt.xlabel("Number of frames per sequence")
        plt.show()

        nMarkers = len(marker_column_names) // 3
        average_shape = get_average_shape(nMarkers)

        df = remove_time_duplicates(df)

        for seqID in df["seqID"].unique():
            (
                times,
                markers,
                Lambda,
                Modes,
                bn,
                Psi,
                phase_shifts,
                dmd_results,
                keypoints,
            ) = run_DMD_sequence(
                seqID,
                df,
                marker_column_names,
                average_shape,
                n_Modes,
                min_seq_length,
                eig_constraints,
                d,
            )
            if times is not None:
                save_sequence_results(
                    save_path,
                    seqID,
                    times,
                    markers,
                    Lambda,
                    Modes,
                    bn,
                    Psi,
                    phase_shifts,
                    dmd_results,
                    keypoints,
                )

    except Exception as e:
        msg = f"Batch DMD analysis failed: {e!s}"
        raise ComputationError(msg) from e


def run_DMD_sequence(
    seqID: str,
    df: pd.DataFrame,
    marker_column_names: np.ndarray,
    average_shape: np.ndarray | None = None,
    n_Modes: int = DEFAULT_N_MODES,
    min_seq_length: int | None = DEFAULT_MIN_SEQ_LENGTH,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    d: int = DEFAULT_D,
    interpolate: bool = False,
) -> tuple[np.ndarray | None, ...]:
    """
    Runs the complete DMD analysis pipeline for a single sequence.

    Args:
        seqID: The ID of the sequence to process
        df: DataFrame containing the full dataset
        marker_column_names: List of column names corresponding to marker coordinates
        average_shape: The mean shape to subtract/add back. If None, it's calculated
        n_Modes: Number of DMD modes
        min_seq_length: Minimum sequence length required
        eig_constraints: Constraints for BOPDMD eigenvalues
        d: Hankel matrix delay parameter
        interpolate: Whether to interpolate missing time points

    Returns:
        Tuple of (times, original_markers_reshaped, Lambda, Modes, bn, Psi,
                  phase_shifts, dmd_results, keypoints)
        Returns None for all if sequence is too short

    Raises:
        ValidationError: If input validation fails
        ComputationError: If DMD computation fails
    """
    try:
        markers, times = load_sequence_data(df, seqID, marker_column_names)

        if average_shape is None:
            nMarkers = len(marker_column_names) // 3
            average_shape = get_average_shape(nMarkers)
        else:
            num_markers = average_shape.shape[1]

        if min_seq_length is None:
            min_seq_length = n_Modes + 1

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
            verbose=False,
        )

        if dmd_results is None:
            return None, None, None, None, None, None, None, None, None

        return (
            times,
            markers,
            Lambda,
            Modes,
            bn,
            Psi,
            phase_shifts,
            dmd_results,
            keypoints,
        )

    except Exception as e:
        msg = f"Error processing sequence {seqID}: {e!s}"
        raise ComputationError(msg) from e
