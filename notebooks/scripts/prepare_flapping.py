"""Prepare binned flapping data for Toothless 9 m straight flights.

Loads raw motion-capture sequences from data/samples/, bins them
into a smooth mean wingbeat (bin size 0.005 s), and saves the
processed markers and times to data/processed/.

Run from the notebooks/ directory:
    python scripts/prepare_flapping.py
"""

import numpy as np
import pandas as pd

from birddmd import bin_dataframe_means, load_sequence_data, remove_time_duplicates

# ── Load raw data ────────────────────────────────────────────────────
column_names = np.load("../data/samples/ColumnNames.npz")
marker_column_names = column_names["marker_column_names"]
info_column_names = column_names["info_column_names"]

file = np.load(
    "../data/samples/Flapping_9mToothless_BilateralNoRot.npz",
    allow_pickle=True,
)
marker_df = pd.DataFrame(file["marker_data"], columns=marker_column_names)
info_df = pd.DataFrame(file["info_data"], columns=info_column_names)
wingbeat_df = pd.concat([info_df, marker_df], axis=1)

# ── Bin ──────────────────────────────────────────────────────────────
binned_df = bin_dataframe_means(
    wingbeat_df,
    "time",
    bin_size=0.005,
    numeric_cast_columns=["HorzDistance", "VertDistance"],
)
binned_df = remove_time_duplicates(binned_df)
binned_df = binned_df[binned_df["time"] >= 0]

markers, times = load_sequence_data(binned_df, "binned", marker_column_names)
markers = markers.reshape(-1, 8, 3)

# ── Save ─────────────────────────────────────────────────────────────
out = "../data/processed/Toothless_flapping_binned.npz"
np.savez(out, markers=markers, times=times)
print(f"Saved {out}: markers {markers.shape}, times {times.shape}")
