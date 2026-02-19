"""Core DMD pipeline tests.

Covers data loading, DMD fitting, result shapes, conjugate-pair
detection, reconstruction, forecasting, RMSE, and API surface.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import BirdDMD
from BirdDMD import (
    DMDResult,
    bin_dataframe_means,
    compute_rmse,
    filter_sequences,
    forecast,
    get_average_shape,
    load_sequence_data,
    normalise_data,
    plot_amplitude_ranking,
    reconstruct,
    remove_time_duplicates,
    run_dmd,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_data():
    """Load the standard Toothless 9m bilateral dataset."""
    col_names = np.load(SAMPLES_DIR / "ColumnNames.npz")
    marker_cols = col_names["marker_column_names"]
    info_cols = col_names["info_column_names"]

    f = np.load(
        SAMPLES_DIR / "Flapping_9mStraightTurnToothless_Bilateral.npz",
        allow_pickle=True,
    )
    marker_df = pd.DataFrame(f["marker_data"], columns=marker_cols)
    info_df = pd.DataFrame(f["info_data"], columns=info_cols)
    wingbeat_df = pd.concat([info_df, marker_df], axis=1)

    return {
        "wingbeat_df": wingbeat_df,
        "marker_column_names": marker_cols,
        "info_column_names": info_cols,
    }


@pytest.fixture(scope="session")
def dmd_result(sample_data):
    """Run DMD on binned data and return a DMDResult."""
    marker_cols = sample_data["marker_column_names"]
    df = sample_data["wingbeat_df"]

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

    return run_dmd(
        data=markers,
        times=times,
        n_modes=6,
        d=2,
        eig_constraints={"conjugate_pairs"},
        n_markers=n_markers,
        average_shape=average_shape,
        verbose=False,
    )


# ── Data loading ────────────────────────────────────────────────────


def test_data_loading(sample_data):
    """Data loads with correct shapes and types."""
    df = sample_data["wingbeat_df"]
    cols = sample_data["marker_column_names"]

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(c in df.columns for c in cols)
    assert df["seqID"].nunique() > 0


def test_remove_time_duplicates(sample_data):
    """Duplicate removal works correctly."""
    df = sample_data["wingbeat_df"]
    cleaned = remove_time_duplicates(df)

    assert not cleaned.duplicated(subset=["frameID"]).any()
    assert 0 < len(cleaned) <= len(df)


# ── DMDResult shape tests ──────────────────────────────────────────


def test_result_is_dataclass(dmd_result):
    """run_dmd returns a DMDResult."""
    assert isinstance(dmd_result, DMDResult)


def test_result_shapes(dmd_result):
    """All result arrays have expected shapes."""
    r = dmd_result
    assert r.n_modes == 6
    assert r.eigenvalues.shape == (6,)
    assert r.amplitudes.shape == (6,)
    assert r.frequencies_hz.shape == (6,)
    assert r.growth_rates.shape == (6,)
    assert r.mode_magnitudes.shape == (8, 3, 6)
    assert r.modes.shape[1] == 6
    assert r.phase_shifts.shape == r.modes.shape
    assert r.reconstruction.shape[1] == 8
    assert r.reconstruction.shape[2] == 3
    assert r.sort_indices.shape == (6,)


def test_conjugate_pairs(dmd_result):
    """Six modes produce three conjugate pairs."""
    assert dmd_result.n_pairs == 3
    for i, j in dmd_result.conjugate_pairs:
        assert i != j, "All modes should be complex conjugate pairs"


def test_pair_indices(dmd_result):
    """pair_indices returns correct mode indices."""
    indices = dmd_result.pair_indices([0, 1])
    assert len(indices) == 4  # Two pairs x 2 modes each


def test_pair_frequency(dmd_result):
    """pair_frequency returns positive Hz values."""
    for p in range(dmd_result.n_pairs):
        freq = dmd_result.pair_frequency(p)
        assert freq >= 0


def test_pair_frequency_uses_unsorted_eigenvalues(dmd_result):
    """pair_frequency uses original eigenvalue ordering, not amplitude-sorted.

    Regression test: pair_frequency previously read from the
    amplitude-sorted frequencies_hz array using conjugate_pairs
    indices that are in the original (unsorted) eigenvalue space.
    This caused pair frequencies to be scrambled.
    """
    dmd = dmd_result._dmd_object
    for p in range(dmd_result.n_pairs):
        i, j = dmd_result.conjugate_pairs[p]
        # Ground truth from original eigenvalues
        expected = float(np.mean(np.abs(np.imag(dmd.eigs[[i, j]]))) / (2 * np.pi))
        actual = dmd_result.pair_frequency(p)
        assert abs(actual - expected) < 1e-10, (
            f"Pair {p}: expected {expected:.4f} Hz, got {actual:.4f} Hz"
        )


# ── Reconstruction ──────────────────────────────────────────────────


def test_reconstruct_all_modes(dmd_result):
    """Full reconstruction has correct shape."""
    recon = reconstruct(dmd_result)
    assert recon.shape[1] == 8
    assert recon.shape[2] == 3


def test_reconstruct_by_pairs(dmd_result):
    """Pair-based reconstruction has correct shape."""
    for p in range(dmd_result.n_pairs):
        recon = reconstruct(dmd_result, pairs=[p])
        assert recon.shape[1] == 8
        assert recon.shape[2] == 3


def test_reconstruct_stable(dmd_result):
    """Stable (phasor) reconstruction works."""
    recon = reconstruct(dmd_result, stable=True)
    assert recon.shape[1] == 8
    assert np.all(np.isfinite(recon))


def test_reconstruct_pairs_vs_modes_exclusive(dmd_result):
    """Cannot supply both pairs and mode_indices."""
    with pytest.raises(ValueError, match="not both"):
        reconstruct(dmd_result, pairs=[0], mode_indices=[0, 1])


# ── Forecast ────────────────────────────────────────────────────────


def test_forecast_extended_time(dmd_result):
    """Forecast produces output for extended time."""
    t = dmd_result.times
    fake_time = np.linspace(t[0], t[-1] * 3, len(t) * 3)
    fc = forecast(dmd_result, fake_time, pairs=[0, 1])

    assert fc.shape == (len(fake_time), 8, 3)
    assert np.all(np.isfinite(fc))


def test_forecast_with_modified_eigenvalues(dmd_result):
    """Forecast with zeroed frequencies works."""
    t = dmd_result.times
    fake_time = np.linspace(t[0], t[-1] * 3, len(t) * 3)
    pairs = dmd_result.conjugate_pairs
    modes_to_zero = list(pairs[0])

    fc = forecast(
        dmd_result,
        fake_time,
        mode_indices=list(pairs[0]) + list(pairs[1]),
        modify_eigenvalues={"zero_modes": modes_to_zero},
    )
    assert fc.shape == (len(fake_time), 8, 3)


def test_forecast_with_changed_freq(dmd_result):
    """Forecast with changed frequency works."""
    t = dmd_result.times
    fake_time = np.linspace(t[0], t[-1] * 3, len(t) * 3)
    pairs = dmd_result.conjugate_pairs
    half_freq = abs(np.imag(dmd_result.eigenvalues[pairs[0][0]])) / 2

    fc = forecast(
        dmd_result,
        fake_time,
        mode_indices=list(pairs[0]),
        modify_eigenvalues={
            "zero_modes": list(pairs[0]),
            "changed_freq": half_freq,
        },
    )
    assert fc.shape == (len(fake_time), 8, 3)


# ── Statistics ──────────────────────────────────────────────────────


def test_compute_rmse(dmd_result):
    """RMSE is non-negative and reasonably small."""
    # Need original markers — re-derive from reconstruction context
    # Use reconstruction vs. slight perturbation as a sanity check
    recon = dmd_result.reconstruction
    noise = recon + np.random.randn(*recon.shape) * 0.001
    rmse = compute_rmse(recon, noise)

    assert rmse.shape[0] == recon.shape[0]
    assert np.all(rmse >= 0)
    assert np.mean(rmse) < 0.01


def test_filter_sequences(sample_data):
    """Sequence filtering returns valid IDs."""
    df = sample_data["wingbeat_df"]
    valid = filter_sequences(df, gap_threshold=0.15, time_start_max=0.15, min_frames=10)
    assert isinstance(valid, list)
    assert len(valid) > 0


# ── Plotting ────────────────────────────────────────────────────────


def test_amplitude_ranking():
    """Amplitude ranking plot runs without error."""
    # We need markers and times — use a minimal synthetic test
    n_frames = 60
    n_markers = 8
    times = np.linspace(0, 1, n_frames)
    markers = np.random.randn(n_frames, n_markers, 3) * 0.01
    avg = np.zeros((1, n_markers, 3))

    def normalise_fn(m):
        return normalise_data(m, avg)

    fig, _ax, amps = plot_amplitude_ranking(
        markers,
        times,
        max_modes=10,
        d=2,
        eig_constraints={"conjugate_pairs"},
        normalise_fn=normalise_fn,
    )
    assert fig is not None
    assert len(amps) > 0
    plt.close(fig)


# ── API surface ─────────────────────────────────────────────────────


def test_version():
    """Version is 1.0.0."""
    assert BirdDMD.__version__ == "1.0.0"


def test_old_functions_removed():
    """Old API functions are not accessible."""
    old_names = [
        "reconstruct_dmd",
        "reconstruct_specific_modes",
        "run_forecast",
        "run_forecast_with_modified_modes",
        "modify_mode_frequencies",
        "reconstruct_conjugate_pairs",
        "reorder_dmd_results",
        "make_dmd_results_dict",
        "add_mode_pair_fft_features",
        "BirdDMDError",
        "ValidationError",
        "ComputationError",
    ]
    for name in old_names:
        assert not hasattr(BirdDMD, name), f"{name} should be removed"


def test_new_functions_accessible():
    """New API functions are accessible."""
    for name in BirdDMD.__all__:
        assert hasattr(BirdDMD, name), f"{name} missing from BirdDMD"
