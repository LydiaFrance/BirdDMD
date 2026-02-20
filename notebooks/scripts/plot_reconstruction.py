"""Plotting functions for notebook 06 — Reconstruction accuracy.

All functions accept a ``results_df`` produced by
``batch_rmse_analysis(per_frame=True)`` and return a matplotlib Figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# ── Constants ────────────────────────────────────────────────────────
HIST_COLOUR = "#66CDAA"
SCATTER_COLOUR = "#35D4BA"
MEDIAN_COLOUR = "#A62639"
MARKER_POSITIONS = ["wingtip", "primary", "secondary", "tailtip"]

BIRD_NAMES = {1: "Drogon", 2: "Rhaegal", 3: "Ruby", 4: "Toothless", 5: "Charmander"}
JUVENILE_YEARS = {1: 2017, 2: 2017, 4: 2017, 5: 2020}  # Ruby always adult


# ── Helpers ──────────────────────────────────────────────────────────


def add_bird_labels(df):
    """Add 'bird_label' column (e.g. 'Toothless J', 'Ruby A') to *df*."""

    def _label(row):
        name = BIRD_NAMES[row["BirdID"]]
        age = "J" if row["Year"] == JUVENILE_YEARS.get(row["BirdID"]) else "A"
        return f"{name} {age}"

    df["bird_label"] = df.apply(_label, axis=1)
    return df


# ── Plot functions ───────────────────────────────────────────────────


def plot_overall_histogram(df, bins=200, ax=None):
    """Histogram of per-frame RMSE pooled across all sequences."""
    all_rmse = np.concatenate(df["rmse_per_frame"].values)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(all_rmse, bins=bins, color=HIST_COLOUR, edgecolor="none")
    ax.set_xlabel("Mean RMSE per frame (m)")
    ax.set_ylabel("Number of frames")
    ax.text(
        0.95,
        0.95,
        f"N frames = {len(all_rmse):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    fig.tight_layout()
    return fig


def plot_marker_histograms_per_frame(df, marker_names, bin_width=0.0075):
    """4x2 grid: per-marker RMSE where each data point is a single frame.

    Requires ``rmse_per_frame_per_marker`` in *df* (from
    ``batch_rmse_analysis(per_frame=True)``).
    """
    name_to_idx = {n: i for i, n in enumerate(marker_names)}

    fig, axes = plt.subplots(4, 2, figsize=(6, 5), sharex=True)
    for i, pos in enumerate(MARKER_POSITIONS):
        for j, side in enumerate(["left", "right"]):
            idx = name_to_idx[f"{side}_{pos}"]
            vals = np.concatenate(
                [arr[:, idx] for arr in df["rmse_per_frame_per_marker"].values]
            )
            edges = np.arange(vals.min(), vals.max() + bin_width, bin_width)
            ax = axes[i, j]
            ax.hist(vals, bins=edges, color="#4D8577", edgecolor="none")

            ax.set_title(f"{side} {pos}", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if "tail" in pos:
                ax.set_ylim([0, 12000])
                ax.set_yticks(np.arange(0, 12100, 2000))
            else:
                ax.set_ylim([0, 8000])
                ax.set_yticks(np.arange(0, 8100, 2000))

    axes[-1, 0].set_xlabel("RMSE per frame (m)")
    axes[-1, 1].set_xlabel("RMSE per frame (m)")
    fig.supylabel("Number of frames", fontsize=10)
    fig.tight_layout()
    return fig


def plot_individual_histograms(df, bin_width=0.005):
    """Per-individual histograms of per-frame RMSE (single-column stack)."""
    labels = df["bird_label"].unique()
    fig, axes = plt.subplots(3, 2, figsize=(6, 6), sharex=True)
    axes = axes.flatten()

    for i, lbl in enumerate(labels):
        frames = np.concatenate(
            df.loc[df["bird_label"] == lbl, "rmse_per_frame"].values
        )
        edges = np.arange(frames.min(), frames.max() + bin_width, bin_width)
        ax = axes[i]
        ax.hist(frames, bins=edges, color=HIST_COLOUR, edgecolor="none")

        ax.set_ylabel(lbl, rotation=0, labelpad=60, fontsize=10, va="center")

        max_count = ax.get_ylim()[1]
        y_top = int(np.ceil(max_count / 100) * 100)
        ax.set_ylim(0, y_top)
        ax.set_yticks(np.arange(0, y_top + 1, 100))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("RMSE per frame (m)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_rmse_over_distance(df, bin_width=0.2):
    """Scatter + boxplot of per-frame RMSE vs horizontal distance."""
    all_dist = -np.concatenate(df["horz_distance"].values)
    all_rmse = np.concatenate(df["rmse_per_frame"].values)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(all_dist, all_rmse, s=1, alpha=0.3, color=HIST_COLOUR)

    edges = np.arange(all_dist.min(), all_dist.max() + bin_width, bin_width)
    bin_idx = np.digitize(all_dist, edges)
    groups, centres = [], []
    for b in range(1, len(edges)):
        vals = all_rmse[bin_idx == b]
        if len(vals) > 0:
            groups.append(vals)
            centres.append((edges[b - 1] + edges[b]) / 2)

    if groups:
        ax.boxplot(
            groups,
            positions=centres,
            widths=bin_width * 0.6,
            manage_ticks=False,
            showfliers=False,
        )

    ax.set_xlabel("Horizontal distance to the perch (m)")
    ax.set_ylabel("RMSE per frame (m)")
    fig.tight_layout()
    return fig


def plot_rmse_distance_subsets(results_df, groups, bin_width=0.20):
    """2x3 panel: scatter + boxplots of RMSE vs distance for filtered subsets.

    Parameters
    ----------
    results_df : DataFrame
        Output of ``batch_rmse_analysis(per_frame=True)``.
    groups : list of (str, dict)
        Each entry is ``(panel_title, column_filters)`` where
        *column_filters* maps column names to required values.
    bin_width : float
        Distance bin width in metres.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharex=False, sharey=True)
    axes = axes.flatten()

    for ax, (title, filters) in zip(axes, groups, strict=False):
        mask = pd.Series(True, index=results_df.index)
        for col, val in filters.items():
            mask &= results_df[col] == val
        df_sub = results_df[mask]

        if df_sub.empty:
            ax.set_title(f"{title} (no data)")
            continue

        all_x = -np.concatenate(df_sub["horz_distance"].values)
        all_y = np.concatenate(df_sub["rmse_per_frame"].values)

        bins = np.arange(all_x.min(), all_x.max() + bin_width, bin_width)
        bin_indices = np.digitize(all_x, bins)

        rmse_by_bin = []
        bin_centers = []
        for b in range(1, len(bins)):
            vals = all_y[bin_indices == b]
            if len(vals) > 0:
                rmse_by_bin.append(vals)
                bin_centers.append((bins[b - 1] + bins[b]) / 2)

        ax.scatter(all_x, all_y, alpha=0.3, s=0.5, color=SCATTER_COLOUR)

        if rmse_by_bin:
            ax.boxplot(
                rmse_by_bin,
                positions=bin_centers,
                widths=bin_width * 0.6,
                manage_ticks=False,
                showfliers=False,
                medianprops={"color": MEDIAN_COLOUR},
            )

        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)

    fig.suptitle("RMSE per frame vs horizontal distance (20 cm bins)", fontsize=14)
    fig.text(0.5, 0.02, "Horizontal distance to perch (m)", ha="center", fontsize=12)
    fig.text(
        0.06, 0.5, "RMSE per frame (m)", va="center", rotation="vertical", fontsize=12
    )
    plt.tight_layout(rect=(0.07, 0.05, 1, 0.95))
    return fig
