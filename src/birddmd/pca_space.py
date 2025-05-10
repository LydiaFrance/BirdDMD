"""
PCA-related analysis functions for BirdDMD.

This module provides functions for PCA projection, coordinate space transformation,
and handling unilateral/bilateral marker data in PCA space.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List
import warnings
warnings.filterwarnings("ignore")

def make_unilateral_keypoints(keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares bilateral keypoint data for unilateral PCA projection.

    Args:
        keypoints: Bilateral keypoint data, shape (n_frames, n_bilateral_markers, 3)
                   Assumes alternating left/right markers (L, R, L, R, ...)

    Returns:
        Tuple of (dmd_reconstruction_flat, left_right_bool) where:
        - dmd_reconstruction_flat is flattened data suitable for PCA (n_frames * 2, n_unilateral_markers * 3)
        - left_right_bool is boolean array indicating left (0) or right (1) side for each row

    Raises:
        ValueError: If input shape is invalid
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
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If input shapes are incompatible
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
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If input shapes are incompatible
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
        
    Raises:
        TypeError: If inputs are not of correct type
        ValueError: If input shapes are incompatible
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
