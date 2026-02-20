"""Prepare binned full-flight and gliding data for Toothless 9 m.

Loads raw full-flight and gliding sequences, bins at 0.005 s,
applies time filters, and saves all four arrays to a single NPZ.

Run from the notebooks/ directory:
    python scripts/prepare_full_flight.py
"""

import numpy as np
import pandas as pd

from birddmd import bin_dataframe_means, load_sequence_data, remove_time_duplicates

# ── Column names ─────────────────────────────────────────────────────
column_names = np.load("../data/samples/ColumnNames.npz")
marker_column_names = column_names["marker_column_names"]
info_column_names = column_names["info_column_names"]

# ── Full flight ──────────────────────────────────────────────────────
file = np.load(
    "../data/samples/Full_9mToothless_BilateralNoRot.npz",
    allow_pickle=True,
)
marker_df = pd.DataFrame(file["marker_data"], columns=marker_column_names)
info_df = pd.DataFrame(file["info_data"], columns=info_column_names)
full_df = pd.concat([info_df, marker_df], axis=1)

binned_full = bin_dataframe_means(
    full_df,
    "time",
    bin_size=0.005,
    numeric_cast_columns=["HorzDistance", "VertDistance"],
)
binned_full = remove_time_duplicates(binned_full)
binned_full = binned_full[(binned_full["time"] >= 0.1) & (binned_full["time"] < 1.35)]

markers_full, times_full = load_sequence_data(
    binned_full,
    "binned",
    marker_column_names,
)
markers_full = markers_full.reshape(-1, 8, 3)

# ── Gliding ──────────────────────────────────────────────────────────
file_g = np.load(
    "../data/samples/Gliding_9mToothless_BilateralNoRot.npz",
    allow_pickle=True,
)
marker_df_g = pd.DataFrame(file_g["marker_data"], columns=marker_column_names)
info_df_g = pd.DataFrame(file_g["info_data"], columns=info_column_names)
glide_df = pd.concat([info_df_g, marker_df_g], axis=1)

binned_glide = bin_dataframe_means(
    glide_df,
    "time",
    bin_size=0.005,
    numeric_cast_columns=["HorzDistance", "VertDistance"],
)
binned_glide = remove_time_duplicates(binned_glide)
binned_glide = binned_glide[
    (binned_glide["time"] > 0.78) & (binned_glide["time"] < 1.35)
]

markers_glide, times_glide = load_sequence_data(
    binned_glide,
    "binned",
    marker_column_names,
)
markers_glide = markers_glide.reshape(-1, 8, 3)

# ── Save ─────────────────────────────────────────────────────────────
out = "../data/processed/Toothless_full_flight_binned.npz"
np.savez(
    out,
    markers_full=markers_full,
    times_full=times_full,
    markers_glide=markers_glide,
    times_glide=times_glide,
)
print(f"Saved {out}:")
print(f"  Full flight: markers {markers_full.shape}, times {times_full.shape}")
print(f"  Gliding:     markers {markers_glide.shape}, times {times_glide.shape}")
