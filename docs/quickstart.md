# Quickstart

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[notebooks]"
```

For documentation development:

```bash
uv pip install -e ".[docs]"
```

## Key concepts

birddmd decomposes motion-capture marker trajectories into a small number
of **Dynamic Modes** — spatially coherent oscillatory patterns, each with
a characteristic frequency and growth/decay rate.

The standard workflow is:

1. **Load** marker data into a `(frames, markers, 3)` array
2. **Normalise** by subtracting the time-averaged shape
3. **Fit** DMD with `run_dmd()`, which returns a `DMDResult` dataclass
4. **Analyse** eigenvalues, modes, and dynamics via the result attributes
5. **Reconstruct** or **forecast** using the fitted model

## First analysis

```python
import numpy as np
import birddmd

# Load hawk data from an NPZ sample file
df, marker_cols = birddmd.load_bird_data("Toothless", "Flapping", perch_distance="9m")
avg_shape = birddmd.get_average_shape(n_markers=8)

# Prepare a single sequence
markers, times = birddmd.load_sequence_data(
    df,
    seqID=df["seqID"].unique()[0],
    marker_column_names=marker_cols,
)

# Normalise
markers_norm = birddmd.normalise_data(
    markers.reshape(-1, 8, 3),
    average_shape=avg_shape,
)

# Run DMD
result = birddmd.run_dmd(
    data=markers_norm,
    times=times,
    n_modes=6,
    d=2,
    eig_constraints={"conjugate_pairs"},
    average_shape=avg_shape,
    n_markers=8,
)

# Inspect the result
print(f"Eigenvalues:  {result.eigenvalues}")
print(f"Frequencies:  {result.frequencies_hz}")
print(f"Conjugate pairs: {result.conjugate_pairs}")
```

## The `DMDResult` dataclass

`run_dmd()` returns a [`DMDResult`](api/core.md) with attributes:

| Attribute | Description |
|-----------|-------------|
| `eigenvalues` | Complex DMD eigenvalues ($\lambda$) |
| `modes` | Spatial DMD modes, shape `(n_coords, n_modes)` |
| `amplitudes` | Mode amplitudes ($b_n$) |
| `frequencies_hz` | Oscillation frequencies in Hz |
| `growth_rates` | Exponential growth/decay rates |
| `mode_magnitudes` | Per-marker mode magnitudes, shape `(n_markers, 3, n_modes)` |
| `phase_shifts` | Phase angles, same shape as `modes` |
| `conjugate_pairs` | List of `(i, j)` index pairs |
| `reconstruction` | Reconstructed data, shape `(frames, markers, 3)` |
| `average_shape` | Mean shape subtracted before fitting |
| `times` | Time vector used for fitting |
| `sort_indices` | Amplitude-based sort order |

## Reconstruction and forecasting

```python
# Reconstruct from a subset of modes
recon = birddmd.reconstruct(result, mode_indices=[0, 1])

# Forecast beyond the training window
future_times = np.linspace(times[-1], times[-1] + 0.5, 50)
future = birddmd.forecast(result, times=future_times)
```

## Next steps

- Browse the [API Reference](api/index.md) for detailed function signatures
- Work through the [Notebook Gallery](notebooks/index.md) for manuscript figures
