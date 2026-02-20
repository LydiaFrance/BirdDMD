# BirdDMD

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

BirdDMD is a Python package for analysing bird flight data using Dynamic Mode Decomposition (DMD). It provides tools for processing motion capture data, performing DMD analysis, and visualising the results.

## Features

- **Data Processing**
  - Load and preprocess motion capture data
  - Handle bilateral marker data (left/right wing markers)
  - Remove time duplicates and interpolate missing data
  - Extract sequences based on flight behavior

- **DMD Analysis**
  - Run DMD on individual sequences or batch process multiple sequences
  - Support for different types of input data (marker coordinates, time series)
  - Configurable number of modes and delay embedding
  - Eigenvalue constraints for robust mode computation

- **Mode Analysis**
  - Reconstruct specific modes
  - Modify mode frequencies
  - Forecast trajectories with modified modes
  - Compare original and reconstructed trajectories

- **Visualisation**
  - Plot marker trajectories in 2D and 3D
  - Visualise mode amplitudes and frequencies
  - Compare original and reconstructed sequences
  - Interactive 3D visualisation of wing movements

## Installation

```bash
python -m pip install birddmd
```

From source:
```bash
git clone https://github.com/LydiaFrance/BirdDMD
cd BirdDMD
python -m pip install .
```

## Usage

### Basic DMD Analysis

```python
from birddmd import run_sequence_dmd

# Run DMD on a specific sequence
times, markers, Lambda, Modes, bn, Psi, phase_shifts, dmd_results, keypoints = run_sequence_dmd(
    bird_name="Toothless",
    perch_dist="9m",
    turn="Straight",
    behaviour="Flapping",
    seqID="04_09_038_1",
    n_modes=10,
    d=2,
    eig_constraints={"conjugate_pairs"}
)
```

### Mode Reconstruction and Modification

```python
from birddmd import reconstruct_specific_modes, run_forecast_with_modified_modes

# Reconstruct specific modes
mode_0_1_keypoints = reconstruct_specific_modes(times, dmd_results, [0,1])

# Forecast with modified modes
keypoints = run_forecast_with_modified_modes(
    dmd_results,
    times,
    average_shape,
    num_markers=8,
    mode_indices_to_zero=[0,1],
    selected_mode_indices=[0,1,2,3,6,7]
)
```

### Upsampling and Forecasting

BirdDMD provides powerful capabilities for upsampling and forecasting wing movements. A key feature is the ability to create stable generative models of flapping motion using just a few carefully selected modes:

```python
from birddmd import run_forecast_with_modified_modes

# Create a longer time vector for forecasting
fake_time = np.linspace(times[0], times[-1]*3, len(times)*3)

# Create a stable generative model using only 3 modes
# The first mode (index 0) is forced to zero frequency for stability
keypoints = run_forecast_with_modified_modes(
    dmd_results,
    fake_time,
    average_shape,
    num_markers=8,
    mode_indices_to_zero=[0, 1],  # Force first mode to zero frequency
    selected_mode_indices=[0, 1, 2, 3, 6, 7]  # Use only 3 modes
)
```

This approach creates a stable generative model that closely matches the original flapping motion. The key insights are:
- Using only 3 modes captures the essential dynamics
- Forcing the first mode to zero frequency ensures stability
- The resulting model can generate continuous flapping motion that closely matches the original data

### Visualisation

```python
from birddmd import plot_markers_overDist, plot_amplitude_ranking

# Plot marker trajectories
plot_markers_overDist(wingbeat_df, marker_column_names, x_axis='time')

# Plot mode amplitude ranking
fig, ax, sorted_amps = plot_amplitude_ranking(
    markers,
    times,
    max_modes=20,
    d=2,
    eig_constraints={"conjugate_pairs"}
)
```

## Data Structure

The package works with motion capture data containing 8 markers:
- Left Wingtip
- Right Wingtip
- Left Primary feather
- Right Primary feather
- Left Secondary feather
- Right Secondary feather
- Left Tailtip
- Right Tailtip

Each marker is represented by 3D coordinates (x, y, z) relative to the bird's center of mass.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LydiaFrance/BirdDMD/workflows/CI/badge.svg
[actions-link]:             https://github.com/LydiaFrance/BirdDMD/actions
[pypi-link]:                https://pypi.org/project/BirdDMD/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/BirdDMD
[pypi-version]:             https://img.shields.io/pypi/v/BirdDMD
<!-- prettier-ignore-end -->
