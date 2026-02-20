"""Prepare binned turning data for Toothless 9 m obstacle flights.

Loads the combined full-flight obstacle file
(Full_9mObstacleToothless_BilateralNoRot.npz) with the 2025-07-25
column names, splits by Turn column into Left and Right, bins each
at 0.005 s, and applies the time filters used in the original
analysis.

Run from the notebooks/ directory:
    python scripts/prepare_turning.py
"""

import numpy as np
import pandas as pd

from birddmd import bin_dataframe_means, load_sequence_data, remove_time_duplicates

# ── Column names ─────────────────────────────────────────────────────
col_names_turn = np.load("../data/samples/2025-07-25_ColumnNames.npz")
marker_column_names = col_names_turn["marker_column_names"]
info_column_names_turn = col_names_turn["info_column_names"]

# ── Left / Right turns from combined full-flight file ────────────────
full = np.load(
    "../data/samples/Full_9mObstacleToothless_BilateralNoRot.npz",
    allow_pickle=True,
)
marker_df = pd.DataFrame(full["marker_data"], columns=marker_column_names)
info_df = pd.DataFrame(full["info_data"], columns=info_column_names_turn)
full_df = pd.concat([info_df, marker_df], axis=1)

# Time-filter bounds per condition (applied *after* binning)
time_filters = {
    "left": (0.0, 1.35, False, False),  # 0 < t < 1.35  (exclusive)
    "right": (0.05, 1.35, False, False),  # 0.05 < t < 1.35  (exclusive)
}

arrays = {}

for key, turn_label in [("left", "Left"), ("right", "Right")]:
    subset = full_df[full_df["Turn"] == turn_label].copy()
    print(f"  {key}: {len(subset)} raw frames")

    binned = bin_dataframe_means(
        subset,
        "time",
        bin_size=0.005,
        numeric_cast_columns=["HorzDistance", "VertDistance"],
    )
    binned = remove_time_duplicates(binned)

    lo, hi, inc_lo, inc_hi = time_filters[key]
    mask = (binned["time"] > lo) & (binned["time"] < hi)
    binned = binned[mask]

    markers, times = load_sequence_data(binned, "binned", marker_column_names)
    markers = markers.reshape(-1, 8, 3)

    arrays[f"markers_{key}"] = markers
    arrays[f"times_{key}"] = times
    print(f"  {key}: {markers.shape[0]} frames after binning + time filter")

# ── Save ─────────────────────────────────────────────────────────────
out = "../data/processed/Toothless_turning_binned.npz"
np.savez(out, **arrays)
print(f"\nSaved {out}")
