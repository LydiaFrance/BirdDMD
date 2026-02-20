# BirdDMD

**Dynamic Mode Decomposition for bird flight analysis.**

![](notebooks/figures/DMD_figure1.png)

BirdDMD is a minimal, scientist-friendly toolkit for decomposing
motion-capture time series into spatially coherent oscillatory modes.
It wraps [PyDMD](https://mathlab.github.io/PyDMD/) with sensible
defaults for biological motion data and provides hawk-specific
convenience functions built on the
[morphing_birds](https://github.com/LydiaFrance/morphing_birds) library for animation.


<img src="notebooks/figures/07_synthetic_flight.gif" width="400"/>

## Architecture

```
BirdDMD
├── core          ← run_dmd(), DMDResult, convergence analysis
├── reconstruction← reconstruct(), forecast()
├── data          ← validation, normalisation, binning, reshaping
├── stats         ← RMSE, variance explained, sequence filtering
├── plotting      ← dataset-agnostic DMD visualisation
├── hawk          ← hawk-specific loading, wrappers, and plots
└── _constants    ← named constants (marker positions, colours)
```

The package separates **dataset-agnostic** DMD utilities (`core`, `data`,
`reconstruction`, `stats`, `plotting`) from **hawk-specific** functions
(`hawk`, `_constants`), so the core analysis can be reused with other
motion-capture datasets.

## Quick links

- [Quickstart guide](quickstart.md) — installation and first analysis
- [API Reference](api/index.md) — full module documentation
- [Notebook Gallery](notebooks/index.md) — worked examples from the manuscript

## Notebooks

| Notebook | Topic |
|----------|-------|
| [00 Introduction](notebooks/00_introduction.ipynb) | DMD theory, synthetic example, and data description |
| [01 Flapping Modes](notebooks/01_flapping_modes.ipynb) | Flapping mode decomposition |
| [03 Double Frequency](notebooks/03_double_frequency.ipynb) | The doubled-frequency mode (2&omega;) |
| [04 Full Flight](notebooks/04_full_flight.ipynb) | Full flight trajectory: flapping to gliding |
| [05 Turning](notebooks/05_turning.ipynb) | Obstacle avoidance and turning |
| [06 Reconstruction Accuracy](notebooks/06_reconstruction_accuracy.ipynb) | Batch RMSE statistics |
| [07 Generative Model](notebooks/07_generative_model.ipynb) | Forecasting, frequency modification, and synthetic flight |

## Standard settings

- **8 bilateral markers** &times; 3 axes = 24 coordinates per frame
- `n_modes=6`, `d=2`, `eig_constraints={"conjugate_pairs"}`
