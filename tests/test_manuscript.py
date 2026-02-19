"""Manuscript number verification tests.

These tests verify that numerical results match the published values
in the manuscript.  They are the critical regression tests — if these
pass, the rewrite is faithful.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from BirdDMD import (
    bin_dataframe_means,
    compute_rmse,
    filter_sequences,
    forecast,
    get_average_shape,
    load_bird_data,
    load_sequence_data,
    remove_time_duplicates,
    run_dmd,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def toothless_9m():
    """Load, bin, and run DMD on Toothless adult 9m straight data."""
    col_names = np.load(SAMPLES_DIR / "ColumnNames.npz")
    marker_cols = col_names["marker_column_names"]
    info_cols = col_names["info_column_names"]

    f = np.load(
        SAMPLES_DIR / "Flapping_9mStraightTurnToothless_Bilateral.npz",
        allow_pickle=True,
    )
    marker_df = pd.DataFrame(f["marker_data"], columns=marker_cols)
    info_df = pd.DataFrame(f["info_data"], columns=info_cols)
    df = pd.concat([info_df, marker_df], axis=1)

    binned = bin_dataframe_means(
        df,
        "time",
        bin_size=0.005,
        numeric_cast_columns=["HorzDistance", "VertDistance"],
    )
    binned = remove_time_duplicates(binned)
    binned = binned[binned["time"] >= 0]

    markers, times = load_sequence_data(binned, "binned", marker_cols)
    n_markers = 8
    markers = markers.reshape(-1, n_markers, 3)
    average_shape = get_average_shape(n_markers)

    result = run_dmd(
        data=markers,
        times=times,
        n_modes=6,
        d=2,
        eig_constraints={"conjugate_pairs"},
        n_markers=n_markers,
        average_shape=average_shape,
        verbose=False,
    )

    return {
        "result": result,
        "markers": markers,
        "times": times,
        "df": df,
        "marker_cols": marker_cols,
        "average_shape": average_shape,
    }


# ── Table 1: Toothless adult frequencies ───────────────────────────


def test_toothless_adult_f1(toothless_9m):
    """Toothless adult wingbeat frequency ≈ 4.51 Hz (Table 1)."""
    result = toothless_9m["result"]
    # Find the wingbeat pair (closest to ~4.5 Hz)
    pair_freqs = [result.pair_frequency(p) for p in range(result.n_pairs)]
    wb_freq = min(pair_freqs, key=lambda f: abs(f - 4.5))
    assert abs(wb_freq - 4.51) < 0.05, f"Expected ~4.51 Hz, got {wb_freq:.2f}"


def test_toothless_adult_f2(toothless_9m):
    """Toothless adult doubled frequency ≈ 8.76 Hz (Table 1)."""
    result = toothless_9m["result"]
    pair_freqs = [result.pair_frequency(p) for p in range(result.n_pairs)]
    fold_freq = min(pair_freqs, key=lambda f: abs(f - 8.8))
    assert abs(fold_freq - 8.76) < 0.05, f"Expected ~8.76 Hz, got {fold_freq:.2f}"


def test_frequency_ratio(toothless_9m):
    """Frequency ratio f2/f1 ≈ 1.94 (Table 1)."""
    result = toothless_9m["result"]
    pair_freqs = [result.pair_frequency(p) for p in range(result.n_pairs)]
    wb = min(pair_freqs, key=lambda f: abs(f - 4.5))
    fold = min(pair_freqs, key=lambda f: abs(f - 8.8))
    ratio = fold / wb
    assert abs(ratio - 1.94) < 0.05, f"Expected ~1.94, got {ratio:.2f}"


# ── Figure 1 caption ───────────────────────────────────────────────


def test_number_of_flights(toothless_9m):
    """Toothless 9m dataset has 67 flights (Figure 1 caption)."""
    df = toothless_9m["df"]
    n_seq = df["seqID"].nunique()
    assert n_seq == 67, f"Expected 67 sequences, got {n_seq}"


def test_rmse_below_threshold(toothless_9m):
    """RMSE ≤ 1.3% of wingspan ≈ 13 mm (Figure 1 caption).

    Wingspan is approximately 1005 mm, so 1.3% ≈ 13 mm = 0.013 m.
    """
    result = toothless_9m["result"]
    markers = toothless_9m["markers"]
    gt = markers[:-1]
    rmse = compute_rmse(result.reconstruction, gt)
    mean_rmse = np.mean(rmse)
    # RMSE should be well below 0.013 m for binned mean data
    assert mean_rmse < 0.013, f"Mean RMSE {mean_rmse:.4f} exceeds 0.013 m"


# ── Section 6: Batch analysis ──────────────────────────────────────


@pytest.fixture(scope="module")
def all_flapping():
    """Load the aggregated flapping dataset (all birds)."""
    col_path = SAMPLES_DIR / "2025-07-25_ColumnNames.npz"
    data_path = SAMPLES_DIR / "FlappingBilateralNoRot.npz"
    if not col_path.exists() or not data_path.exists():
        pytest.skip("Aggregated flapping dataset not available")

    col_names = np.load(col_path)
    marker_cols = col_names["marker_column_names"]
    info_cols = col_names["info_column_names"]

    f = np.load(data_path, allow_pickle=True)
    marker_df = pd.DataFrame(f["marker_data"], columns=marker_cols)
    info_df = pd.DataFrame(f["info_data"], columns=info_cols)
    df = pd.concat([info_df, marker_df], axis=1)
    return remove_time_duplicates(df)


def test_total_filtered_sequences(all_flapping):
    """413 sequences pass quality filter (Section 6)."""
    valid = filter_sequences(
        all_flapping,
        gap_threshold=0.15,
        time_start_max=0.15,
        min_frames=30,
    )
    assert len(valid) == 413, f"Expected 413, got {len(valid)}"


def test_total_frames(all_flapping):
    """18,676 total frames across all filtered sequences (Section 6)."""
    valid = filter_sequences(
        all_flapping,
        gap_threshold=0.15,
        time_start_max=0.15,
        min_frames=30,
    )
    total = sum(len(all_flapping[all_flapping["seqID"] == s]) for s in valid)
    assert total == 18676, f"Expected 18676, got {total}"


# ── Figure 5 caption: turning flights ──────────────────────────────


def test_left_turns():
    """36 left-turn sequences (Figure 5 caption)."""
    try:
        df, _ = load_bird_data(
            "Toothless",
            "Flapping",
            perch_distance="9m",
            turn="Left",
        )
    except FileNotFoundError:
        pytest.skip("Left-turn data not available")

    df = remove_time_duplicates(df)
    n_seq = df["seqID"].nunique()
    assert n_seq == 36, f"Expected 36, got {n_seq}"


def test_right_turns():
    """30 right-turn sequences (Figure 5 caption)."""
    try:
        df, _ = load_bird_data(
            "Toothless",
            "Flapping",
            perch_distance="9m",
            turn="Right",
        )
    except FileNotFoundError:
        pytest.skip("Right-turn data not available")

    df = remove_time_duplicates(df)
    n_seq = df["seqID"].nunique()
    assert n_seq == 30, f"Expected 30, got {n_seq}"


# ── Section 7: Stabilised forecast ──────────────────────────────────


def test_stabilised_forecast_bounded(toothless_9m):
    """Stabilised forecast stays bounded over 5 seconds (Section 7).

    Per NB07, the stabilised forecast uses only the oscillatory pairs
    (wingbeat + folding), not the base pair, to avoid the near-zero
    frequency mode which adds a slowly varying offset.
    """
    result = toothless_9m["result"]
    long_time = np.linspace(0, 5.0, 1000)

    # Identify the oscillatory pairs (non-zero frequency)
    oscillatory = [
        p
        for p in range(result.n_pairs)
        if result.pair_frequency(p) > 1.0  # Hz
    ]

    fc = forecast(result, long_time, pairs=oscillatory, stable=True)
    assert np.all(np.isfinite(fc)), "Forecast contains NaN or Inf"

    # Phasor output should be periodic: max deviation in the last
    # second should not exceed the first second (plus tolerance).
    frames_per_sec = len(long_time) // 5
    first_sec = fc[:frames_per_sec]
    last_sec = fc[-frames_per_sec:]
    max_first = np.max(np.abs(first_sec))
    max_last = np.max(np.abs(last_sec))
    assert max_last <= max_first * 1.05, (
        f"Forecast grew: max first sec = {max_first:.3f}, max last sec = {max_last:.3f}"
    )
