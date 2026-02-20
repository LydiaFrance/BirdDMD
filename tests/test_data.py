"""Data handling tests.

Covers validation, normalisation, binning, sequence extraction,
and expansion utilities.
"""

import numpy as np
import pandas as pd
import pytest

from birddmd.data import (
    add_average_shape,
    bin_dataframe_means,
    expand_marker_sequence,
    expand_time_sequence,
    load_sequence_data,
    normalise_data,
    remove_time_duplicates,
    validate_marker_data,
)
from birddmd.stats import variance_explained

# ── Validation ──────────────────────────────────────────────────────


def test_validate_3d():
    """3-D input is flattened correctly."""
    data = np.random.randn(50, 8, 3)
    flat, nf, nm, nc = validate_marker_data(data)
    assert flat.shape == (50, 24)
    assert nf == 50
    assert nm == 8
    assert nc == 24


def test_validate_2d():
    """2-D input passes through."""
    data = np.random.randn(50, 24)
    flat, _nf, nm, _nc = validate_marker_data(data)
    assert flat.shape == (50, 24)
    assert nm == 8


def test_validate_bad_shape():
    """1-D input raises ValueError."""
    with pytest.raises(ValueError, match="must be 2-D or 3-D"):
        validate_marker_data(np.random.randn(100))


def test_validate_bad_spatial():
    """3-D input with wrong last axis raises ValueError."""
    with pytest.raises(ValueError, match="Expected 3 coordinates"):
        validate_marker_data(np.random.randn(50, 8, 4))


def test_validate_not_divisible():
    """2-D with non-divisible-by-3 columns raises ValueError."""
    with pytest.raises(ValueError, match="must be divisible by 3"):
        validate_marker_data(np.random.randn(50, 25))


# ── Normalisation ───────────────────────────────────────────────────


def test_normalise_roundtrip_3d():
    """normalise + add_average_shape is identity."""
    data = np.random.randn(50, 8, 3)
    avg = np.random.randn(1, 8, 3)
    centred = normalise_data(data, avg)
    recovered = add_average_shape(centred, avg)
    np.testing.assert_allclose(recovered, data)


def test_normalise_roundtrip_2d():
    """2-D normalise roundtrip."""
    data = np.random.randn(50, 24)
    avg = np.random.randn(1, 24)
    centred = normalise_data(data, avg)
    recovered = add_average_shape(centred, avg)
    np.testing.assert_allclose(recovered, data)


# ── Binning ─────────────────────────────────────────────────────────


def test_bin_dataframe_means():
    """Binning produces fewer rows with seqID='binned'."""
    df = pd.DataFrame(
        {
            "time": np.linspace(0, 1, 100),
            "x": np.random.randn(100),
            "seqID": "seq1",
        }
    )
    binned = bin_dataframe_means(df, "time", bin_size=0.1)
    assert len(binned) < len(df)
    assert (binned["seqID"] == "binned").all()


# ── Expansion ───────────────────────────────────────────────────────


def test_expand_time_sequence():
    """Expanded time is longer than original."""
    t = np.linspace(0, 1, 50)
    expanded = expand_time_sequence(t, expansion_factor=3.0)
    assert len(expanded) == 150
    assert expanded[-1] == pytest.approx(3.0)


def test_expand_marker_sequence():
    """Expanded markers match expanded time length."""
    t = np.linspace(0, 1, 50)
    m = np.random.randn(50, 24)
    et = expand_time_sequence(t, 2.0)
    em = expand_marker_sequence(t, m, et)
    assert em.shape == (len(et), 24)


# ── Sequence extraction ─────────────────────────────────────────────


def test_load_sequence_data():
    """Extracting a sequence by ID works."""
    df = pd.DataFrame(
        {
            "seqID": ["a"] * 10 + ["b"] * 5,
            "time": list(range(10)) + list(range(5)),
            "x": np.random.randn(15),
            "y": np.random.randn(15),
        }
    )
    markers, times = load_sequence_data(df, "a", np.array(["x", "y"]))
    assert markers.shape == (10, 2)
    assert times.shape == (10,)


def test_remove_time_duplicates():
    """Duplicate rows are removed."""
    df = pd.DataFrame(
        {
            "frameID": [1, 2, 2, 3, 4],
            "value": [10, 20, 21, 30, 40],
        }
    )
    cleaned = remove_time_duplicates(df)
    assert len(cleaned) == 4
    assert not cleaned.duplicated(subset=["frameID"]).any()


# ── Variance explained ──────────────────────────────────────────────


def test_variance_explained_perfect():
    """Perfect reconstruction gives VE = 1.0."""
    data = np.random.randn(50, 8, 3)
    ve = variance_explained(data, data)
    assert ve == pytest.approx(1.0)


def test_variance_explained_zero():
    """Zero reconstruction gives VE ≈ negative."""
    data = np.random.randn(50, 8, 3) + 10  # shift away from zero
    zeros = np.zeros_like(data)
    ve = variance_explained(data, zeros)
    assert ve < 0  # Worse than mean model
