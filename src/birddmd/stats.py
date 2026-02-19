"""Statistical analysis utilities.

Provides the canonical RMSE implementation, sequence quality filtering,
and variance-explained computation.

Functions
---------
compute_rmse
    Per-frame root mean square error.
filter_sequences
    Return sequence IDs that pass quality criteria.
variance_explained
    Fraction of variance captured by a reconstruction (R²-like).
"""

import numpy as np
import pandas as pd

from ._constants import (
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_MIN_FRAMES,
    DEFAULT_TIME_START_MAX,
)


def compute_rmse(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Per-frame RMSE between *reconstruction* and *ground_truth*.

    Both arrays must share the same shape, typically
    ``(n_frames, n_markers, 3)``.

    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed data.
    ground_truth : np.ndarray
        Original data to compare against.

    Returns
    -------
    np.ndarray
        RMSE per frame, shape ``(n_frames,)``.

    Notes
    -----
    Computed as:

        RMSE(t) = √( mean( (x̂(t) - x(t))² ) )

    where the mean is taken over all markers and coordinates at each
    time step.
    """
    return np.sqrt(np.mean((reconstruction - ground_truth) ** 2, axis=(1, 2)))


def filter_sequences(
    df: pd.DataFrame,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    time_start_max: float = DEFAULT_TIME_START_MAX,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> list[str]:
    """Return sequence IDs that pass quality filters.

    Sequences are rejected if they contain time gaps larger than
    *gap_threshold*, start later than *time_start_max*, or have fewer
    than *min_frames* frames.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``seqID``, ``time``, and ``frameID`` columns.
    gap_threshold : float
        Maximum allowed gap between consecutive frames (seconds).
    time_start_max : float
        Maximum allowed start time (seconds).
    min_frames : int
        Minimum required frame count.

    Returns
    -------
    list of str
        Sequence IDs passing all filters.
    """
    sorted_df = df.sort_values(by=["seqID", "time"]).copy()
    sorted_df["time_diff"] = sorted_df.groupby("seqID")["time"].diff()

    stats = (
        sorted_df.groupby("seqID")
        .agg(
            total_frames=("frameID", "count"),
            max_gap=("time_diff", "max"),
            min_time=("time", "min"),
        )
        .reset_index()
    )

    passes = (
        (stats["max_gap"] <= gap_threshold)
        & (stats["min_time"] <= time_start_max)
        & (stats["total_frames"] >= min_frames)
    )
    return stats.loc[passes, "seqID"].tolist()


def variance_explained(
    original: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Fraction of variance captured by *reconstruction*.

    Analogous to R²:

        VE = 1 - SS_res / SS_tot

    Parameters
    ----------
    original : np.ndarray
        Ground-truth data (any shape).
    reconstruction : np.ndarray
        Reconstructed data (same shape as *original*).

    Returns
    -------
    float
        Variance explained, in [0, 1] for a good fit.
    """
    original_flat = original.reshape(-1, original.shape[-1])
    recon_flat = reconstruction.reshape(-1, reconstruction.shape[-1])

    ss_res = np.sum((original_flat - recon_flat) ** 2)
    ss_tot = np.sum((original_flat - original_flat.mean(axis=0)) ** 2)

    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
