# Notebook Gallery

These notebooks reproduce the figures and analyses from the manuscript.
Each notebook is rendered with its saved cell outputs — no re-execution
required.

## Prerequisites

Install the notebook extras to run these interactively:

```bash
uv pip install -e ".[notebooks]"
jupyter lab notebooks/
```

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 00 | [Introduction](00_introduction.ipynb) | DMD theory, synthetic example, and data description |
| 01 | [Flapping Modes](01_flapping_modes.ipynb) | Flapping mode decomposition |
| 03 | [Double Frequency](03_double_frequency.ipynb) | The doubled-frequency mode (2&omega;) |
| 04 | [Full Flight](04_full_flight.ipynb) | Full flight trajectory: flapping to gliding |
| 05 | [Turning](05_turning.ipynb) | Obstacle avoidance and turning |
| 06 | [Reconstruction Accuracy](06_reconstruction_accuracy.ipynb) | Batch RMSE statistics |
| 07 | [Generative Model](07_generative_model.ipynb) | Forecasting, frequency modification, and synthetic flight |
