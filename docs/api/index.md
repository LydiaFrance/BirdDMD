# API Reference

BirdDMD is organised into dataset-agnostic modules and a hawk-specific
convenience layer.

## Modules

| Module | Description |
|--------|-------------|
| [`core`](core.md) | `DMDResult`, `run_dmd()`, conjugate-pair detection, convergence analysis |
| [`data`](data.md) | Validation, normalisation, binning, and reshaping |
| [`reconstruction`](reconstruction.md) | `reconstruct()` and `forecast()` from fitted DMD models |
| [`stats`](stats.md) | RMSE, variance explained, and sequence filtering |
| [`plotting`](plotting.md) | Dataset-agnostic DMD visualisation |
| [`hawk`](hawk.md) | Hawk-specific loading, wrappers, and plots |
| [`_constants`](constants.md) | Named constants (marker positions, colours, metadata columns) |
