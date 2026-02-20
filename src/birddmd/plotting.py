"""Dataset-agnostic DMD visualisation.

Hawk-specific marker plots live in `hawk`.  This module provides
general-purpose DMD plots: amplitude ranking, mode dynamics,
cumulative RMSE comparison, and convergence analysis.

Functions
---------
plot_amplitude_ranking
    Scree-style amplitude vs. rank plot.
plot_mode_dynamics
    Temporal dynamics stacked by conjugate pair.
plot_cumulative_rmse
    Bar chart of cumulative RMSE as pairs are added.
plot_convergence
    RMSE and variance explained vs. mode count.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing

if TYPE_CHECKING:
    from .core import DMDResult


# ── Shared helpers ──────────────────────────────────────────────────


def _remove_spines(ax, keep_bottom: bool = False):
    """Strip spines and ticks from an axes."""
    for spine in ("top", "right", "left", "bottom"):
        if spine == "bottom" and keep_bottom:
            continue
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)


# ── Amplitude ranking ───────────────────────────────────────────────


def plot_amplitude_ranking(
    markers: np.ndarray,
    times: np.ndarray,
    max_modes: int = 20,
    d: int = 2,
    eig_constraints: set | None = None,
    figsize: tuple[int, int] = (6, 2),
    normalise_fn: Callable | None = None,
) -> tuple[Figure, Axes, np.ndarray]:
    """Plot DMD mode amplitudes vs. rank.

    Fits a standalone BOPDMD with *max_modes* modes and plots the
    sorted absolute amplitudes.  Useful for choosing the SVD rank.

    Parameters
    ----------
    markers : np.ndarray
        Marker data (any shape accepted by ``validate_marker_data``).
    times : np.ndarray
        Corresponding time vector.
    max_modes : int
        Number of modes to fit.
    d : int
        Hankel delay.
    eig_constraints : set
        Passed to BOPDMD.
    figsize : tuple
        Figure size.
    normalise_fn : callable, optional
        ``f(markers) -> centred_markers``.  If ``None``, data is
        used as-is (caller should pre-centre).

    Returns
    -------
    fig : Figure
    ax : Axes
    sorted_amplitudes : np.ndarray
    """
    if eig_constraints is None:
        eig_constraints = {"conjugate_pairs"}

    if times.shape[0] != markers.shape[0]:
        msg = "times and markers must have the same frame count"
        raise ValueError(msg)

    required = max_modes + d
    if markers.shape[0] <= required:
        msg = (
            f"Not enough frames ({markers.shape[0]}) for {max_modes} modes "
            f"with d={d}. Need > {required}."
        )
        raise ValueError(msg)

    if normalise_fn is not None:
        markers = normalise_fn(markers)
    flat = markers.reshape(markers.shape[0], -1).T

    print(f"Fitting DMD with max_modes={max_modes}, d={d}...")
    dmd = hankel_preprocessing(
        BOPDMD(svd_rank=max_modes, eig_constraints=eig_constraints),
        d=d,
    )
    dmd.fit(flat, t=times[1:])  # type: ignore[union-attr]
    print("DMD fit complete.")

    abs_amps = np.abs(dmd.amplitudes)  # type: ignore[call-overload]
    sorted_amps = np.sort(abs_amps)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(np.arange(len(sorted_amps)), sorted_amps, marker="o", s=15)
    ax.set_ylabel(r"Amplitude $|\beta|$")
    ax.set_xlabel("DMD Mode Rank (Sorted by Amplitude)")
    ax.set_title(f"DMD Mode Amplitude Ranking (max_modes={max_modes})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=8)
    fig.tight_layout()

    return fig, ax, sorted_amps


# ── Mode dynamics ───────────────────────────────────────────────────


def plot_mode_dynamics(
    times: np.ndarray,
    result: DMDResult,
    title_prefix: str = "",
    x_lim: tuple[float, float] | None = None,
    y_lim: float | list[float] | None = None,
    axes_visible: bool = True,
) -> Figure:
    """Plot temporal dynamics stacked by conjugate pair.

    Parameters
    ----------
    times : np.ndarray
        Time vector (typically ``result.times[1:]``).
    result : DMDResult
        DMD analysis result.
    title_prefix : str
        Prepended to each subplot title.
    x_lim : tuple of float, optional
        ``(xmin, xmax)`` for all subplots.
    y_lim : float or list of float, optional
        Symmetric y-limits.  A list sets per-pair limits.
    axes_visible : bool
        Whether to show grid/labels on all subplots.

    Returns
    -------
    Figure
    """
    dmd = result._dmd_object
    pairs = result.conjugate_pairs
    n_pairs = len(pairs)
    frequencies = np.imag(dmd.eigs) / (2 * np.pi)

    fig = plt.figure(figsize=(4, 2 * n_pairs))

    for idx, (ii, jj) in enumerate(pairs):
        ax = plt.subplot(n_pairs, 1, idx + 1)
        if ii == jj:
            ax.plot(
                times, dmd.dynamics[ii].real, linewidth=2, label=f"Mode {ii} (Real)"
            )
            title = f"Pair {idx}: Real Mode {ii} Frequency: {frequencies[ii]:.2f} Hz"
        else:
            ax.plot(
                times, dmd.dynamics[ii].real, linewidth=4, label=f"Mode {ii} (Primary)"
            )
            ax.plot(
                times,
                dmd.dynamics[jj].real,
                linewidth=4,
                linestyle="-",
                label=f"Mode {jj} (Conjugate)",
            )
            title = f"Pair {idx + 1} ({ii},{jj}) Frequency: {frequencies[ii]:.2f} Hz"
            if axes_visible:
                ax.set_title(f"{title_prefix} {title}", fontsize=10)

        freq_ii = frequencies[ii]
        title += f" | f={freq_ii:.2f} Hz"
        title += f" | \u03bb={dmd.eigs[ii]:.3f}"
        print(title)

        if y_lim is not None:
            lim = (
                float(y_lim[idx])
                if isinstance(y_lim, (list, np.ndarray)) and idx < len(y_lim)
                else float(y_lim)
            )
            if lim is not None:
                lim_val = lim
                ax.set_ylim(-lim_val, lim_val)
        if x_lim is not None:
            ax.set_xlim(x_lim)
            ax.set_xticks(list(x_lim))
            ax.set_xticklabels([str(v) for v in x_lim])

        if not axes_visible:
            _remove_spines(ax)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            if idx == n_pairs - 1:
                ax.xaxis.set_visible(True)
                ax.set_xlabel("time (s)")
                ax.spines["bottom"].set_visible(True)
        else:
            ax.grid(True, alpha=0.3)
            plt.ylabel("Amplitude")
        plt.xlabel("time (s)")

    plt.tight_layout()
    plt.show()
    return fig


# ── Convergence ─────────────────────────────────────────────────────


def plot_convergence(results: dict) -> Figure:
    """Plot RMSE and variance explained vs. mode count.

    Parameters
    ----------
    results : dict
        Output of `convergence_analysis`.

    Returns
    -------
    Figure
    """
    n_modes = np.array(results["n_modes"])
    rmse_mean = np.array(results["rmse_mean"])
    rmse_std = np.array(results["rmse_std"])
    var_exp = np.array(results["variance_explained"])

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].errorbar(
        n_modes,
        rmse_mean,
        yerr=rmse_std,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=4,
        capsize=1,
        capthick=1,
        color="steelblue",
        ecolor="lightblue",
    )
    axes[0].set_xlabel("Number of Modes", fontsize=12)
    axes[0].set_ylabel("Mean RMSE (m)", fontsize=12)
    axes[0].set_title("RMSE", fontsize=9, fontweight="bold")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].set_xticks(n_modes)

    axes[1].plot(
        n_modes,
        var_exp * 100,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=4,
        color="darkgreen",
    )
    axes[1].set_xlabel("Number of Modes", fontsize=12)
    axes[1].set_ylabel("Variance Explained (%)", fontsize=12)
    axes[1].set_title("Variance Explained", fontsize=9, fontweight="bold")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_xticks(n_modes)
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    return fig
