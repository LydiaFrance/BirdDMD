"""
Functions for reconstructing and modifying DMD results.

This module provides functions for reconstructing time series data from DMD modes,
modifying mode frequencies, and handling mode reordering. It includes utilities
for both full reconstruction and selective mode reconstruction.
"""

import numpy as np
from typing import Optional, Tuple, Union, Any, List
import warnings
warnings.filterwarnings("ignore")

from .data_handling import get_average_shape, reshape_data

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
        
    Raises:
        ValueError: If input arrays have incompatible shapes
        TypeError: If inputs are not numpy arrays
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

def reconstruct_specific_modes(times: np.ndarray,
                             dmd_results: Any,
                             mode_indices: Union[List[int], np.ndarray],
                             n_markers: int = 8) -> np.ndarray:
    """Reconstruct specific DMD modes using original DMD outputs.
    
    Args:
        times: Time points array
        dmd_results: DMD object with eigs, modes, amplitudes
        mode_indices: List/array of mode indices to include
        n_markers: Number of markers (default=8)
        
    Returns:
        reconstruction: Reconstructed keypoints (n_frames, n_markers, 3)
        
    Raises:
        ValueError: If dmd_results is missing required attributes
        TypeError: If inputs are not of correct type
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

def modify_mode_frequencies(omega: np.ndarray,
                          Psi: np.ndarray,
                          amplitudes: np.ndarray,
                          mode_indices_to_zero: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a copy of the DMD components with modified frequencies for specified modes.
    
    Args:
        omega: Array of frequencies (imaginary parts of eigenvalues)
        Psi: Complex DMD modes
        amplitudes: Complex DMD amplitudes
        mode_indices_to_zero: List/array of mode indices to set to zero frequency
        
    Returns:
        Tuple of (modified_omega, Psi, amplitudes) where modified_omega has zero frequencies
        for the specified modes while preserving conjugate pair structure
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If mode indices are invalid
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

def run_forecast(dmd_results: Any,
                times: np.ndarray,
                average_shape: np.ndarray,
                num_markers: int) -> np.ndarray:
    """Reconstructs the full trajectory using all computed DMD modes.
    
    Uses the `reconstruct_dmd` function based on phasor notation and adds back the mean shape.
    
    Args:
        dmd_results: Fitted pydmd object containing modes, eigenvalues, amplitudes
        times: Time vector for reconstruction
        average_shape: Mean shape to add back
        num_markers: Number of markers
        
    Returns:
        keypoints: Reconstructed trajectory reshaped to (n_frames, n_markers, 3)
        
    Raises:
        ValueError: If there is a shape mismatch between reconstruction and average shape
    """
    n_coords = num_markers * 3
    
    # Pass the imaginary part of eigenvalues as omega (frequencies)
    reconstruction = reconstruct_dmd(
        times,
        np.imag(dmd_results.eigs),  # Pass frequencies
        dmd_results.modes[:n_coords,:],
        dmd_results.amplitudes
    )

    # Add mean shape and reshape
    # Ensure reconstruction shape is (n_times, n_coords) before adding mean shape
    average_shape = average_shape.reshape(1, -1)
    if reconstruction.shape[1] != average_shape.shape[1]:
        raise ValueError(f"Shape mismatch: Reconstruction ({reconstruction.shape}) vs Average Shape ({average_shape.shape})")
    
    forecast_plus_mean = reconstruction + average_shape  # Broadcasting
    keypoints = reshape_data(forecast_plus_mean, -1, num_markers, 3)

    return keypoints

def run_forecast_with_modified_modes(dmd_results: Any,
                                   times: np.ndarray,
                                   average_shape: np.ndarray,
                                   num_markers: int,
                                   mode_indices_to_zero: Optional[List[int]] = None,
                                   selected_mode_indices: Optional[List[int]] = None) -> np.ndarray:
    """Reconstructs the trajectory using selected DMD modes, with optional frequency modification.
    
    Can both modify frequencies of specific modes and select which modes to include.
    
    Args:
        dmd_results: Fitted pydmd object containing modes, eigenvalues, amplitudes
        times: Time vector for reconstruction
        average_shape: Mean shape to add back
        num_markers: Number of markers
        mode_indices_to_zero: List/array of mode indices to set to zero frequency.
                            If None, no frequencies are modified
        selected_mode_indices: List/array of mode indices to include in reconstruction.
                             If None, all modes are included
        
    Returns:
        keypoints: Reconstructed trajectory reshaped to (n_frames, n_markers, 3)
    """
    n_coords = num_markers * 3
    
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
