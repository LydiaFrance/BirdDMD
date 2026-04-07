"""Hawk-specific data loading, DMD wrappers, and visualisation.

Everything specific to the hawk flight dataset lives here: NPZ file
loading, mean-shape retrieval via ``morphing_birds.Animal3D``, batch RMSE
analysis, and marker-position plots.  General-purpose DMD logic is in
the other modules; this layer fills in hawk defaults.

Functions
---------
load_bird_data
    Load a hawk NPZ file into a DataFrame.
get_average_shape
    Return the mean hawk shape via ``Animal3D``.
normalise_hawk_data
    Centre hawk markers by subtracting the mean shape.
bin_hawk_dataframe
    Bin with hawk-specific column casting.
run_hawk_dmd
    ``run_dmd`` with hawk defaults.
run_sequence_dmd
    Load one hawk sequence and run DMD.
batch_rmse_analysis
    DMD + RMSE on many sequences.
plot_hawk_markers
    4x3 scatter grid of marker positions.
plot_hawk_markers_shaded
    4x3 binned mean +/- 1 SD bands.
plot_hawk_2d
    2x4 y-z projections.
plot_single_sequence
    Single-sequence scatter.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from morphing_birds import Animal3D

from ._constants import (
    HAWK_META_COLUMNS,
    MARKER_COLOURS,
    MARKER_POSITIONS,
    N_BILATERAL_MARKERS,
    N_UNILATERAL_MARKERS,
)
from .core import run_dmd
from .data import (
    bin_dataframe_means,
    load_sequence_data,
    normalise_data,
    remove_time_duplicates,
    spline_interpolation,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .core import DMDResult

# ── Paths ───────────────────────────────────────────────────────────

SAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples")
)

MEAN_SHAPE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mean_hawk_shape.csv"
)


# ── Data loading ────────────────────────────────────────────────────


def load_bird_data(
    bird_name: str,
    behaviour: str,
    perch_distance: str | None = None,
    bilateral: str = "Bilateral",
    turn: str = "Straight",
    verbose: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load hawk marker and info data from NPZ files.

    Constructs a file path from the identifiers, loads the marker and
    info arrays from the NPZ archive, and merges them into a single
    DataFrame.  Column names are read from ``ColumnNames.npz`` in the
    same directory.

    Parameters
    ----------
    bird_name : str
        Individual hawk identifier, e.g. ``'Toothless'``.
    behaviour : str
        Flight phase label used in the filename, e.g. ``'Initial'``,
        ``'Flapping'``.
    perch_distance : str, optional
        Distance to perch, e.g. ``'9m'``.  If ``None`` the distance
        token is omitted from the filename.
    bilateral : str
        ``'Bilateral'`` (8 markers, left + right) or ``'Unilateral'``
        (4 markers, right side only).
    turn : str
        Turn direction: ``'Straight'``, ``'Left'``, or ``'Right'``.
        Only used when *perch_distance* is ``'9m'``.
    verbose : bool
        If ``True``, print the DataFrame shape and sequence count.

    Returns
    -------
    wingbeat_df : pd.DataFrame
        Combined info + marker DataFrame with one row per frame.
        Info columns include ``seqID``, ``frameID``, ``time``, etc.;
        marker columns follow the ``{side}_{position}_{axis}`` naming
        convention.
    marker_column_names : np.ndarray
        1-D string array of marker column names, length
        ``n_markers * 3``.
    """
    pd_str = "" if perch_distance is None else perch_distance

    if perch_distance == "9m":
        path = (
            f"{SAMPLES_DIR}/{behaviour}_{pd_str}{turn}Turn{bird_name}_{bilateral}.npz"
        )
    else:
        path = f"{SAMPLES_DIR}/{behaviour}_{pd_str}{bird_name}_{bilateral}.npz"

    file = np.load(path, allow_pickle=True)
    col_names = np.load(f"{SAMPLES_DIR}/ColumnNames.npz")
    marker_cols = col_names["marker_column_names"]
    info_cols = col_names["info_column_names"]

    marker_df = pd.DataFrame(file["marker_data"], columns=marker_cols)
    info_df = pd.DataFrame(file["info_data"], columns=info_cols)
    wingbeat_df = pd.concat([info_df, marker_df], axis=1)

    if verbose:
        n_seq = wingbeat_df["seqID"].nunique()
        print(
            f"Loaded {bird_name} {bilateral.lower()}: "
            f"{wingbeat_df.shape}, {n_seq} sequences"
        )

    return wingbeat_df, marker_cols


def get_average_shape(
    n_markers: int,
    mean_shape_path: str | None = None,
) -> np.ndarray:
    """Return the mean hawk shape via ``Animal3D``.

    Instantiates a ``morphing_birds.Animal3D`` object from the mean-shape
    CSV and returns either the full bilateral marker set or the
    right-side-only subset.

    Parameters
    ----------
    n_markers : int
        Number of anatomical markers: 8 for bilateral (both wings) or
        4 for unilateral (right side only).
    mean_shape_path : str, optional
        Path to the mean-shape CSV file.  Defaults to
        ``data/mean_hawk_shape.csv`` shipped with the package.

    Returns
    -------
    np.ndarray
        Shape ``(1, n_markers, 3)`` — the time-averaged marker
        positions in 3-D space.

    Raises
    ------
    ValueError
        If *n_markers* is not 4 or 8.
    """
    if mean_shape_path is None:
        mean_shape_path = MEAN_SHAPE_PATH

    hawk3d = Animal3D("hawk", mean_shape_path)
    if n_markers == N_BILATERAL_MARKERS:
        return hawk3d.markers
    if n_markers == N_UNILATERAL_MARKERS:
        return hawk3d.right_markers
    msg = (
        f"Expected {N_UNILATERAL_MARKERS} or {N_BILATERAL_MARKERS} markers,"
        f" got {n_markers}"
    )
    raise ValueError(msg)


def normalise_hawk_data(
    markers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre hawk markers by subtracting the mean hawk shape.

    Infers the marker count from the array shape, loads the
    corresponding mean shape via `get_average_shape`, and
    subtracts it from every frame.

    Parameters
    ----------
    markers : np.ndarray
        Raw marker positions, either shape ``(n_frames, n_markers, 3)``
        or flattened as ``(n_frames, n_markers * 3)``.

    Returns
    -------
    normalised : np.ndarray
        Mean-subtracted markers, same shape as *markers*.
    average_shape : np.ndarray
        The mean shape that was subtracted, shape
        ``(1, n_markers, 3)``.

    Raises
    ------
    ValueError
        If *markers* is not 2-D or 3-D.
    """
    DIMS_2 = 2
    DIMS_3 = 3
    if markers.ndim == DIMS_3:
        n_markers = markers.shape[1]
    elif markers.ndim == DIMS_2:
        n_markers = markers.shape[1] // DIMS_3
    else:
        msg = f"Expected 2-D or 3-D data, got {markers.ndim}-D"
        raise ValueError(msg)

    average_shape = get_average_shape(n_markers)
    return normalise_data(markers, average_shape), average_shape


# ── Hawk-specific binning ───────────────────────────────────────────


def bin_hawk_dataframe(
    dataframe: pd.DataFrame,
    x_axis: str = "HorzDistance",
    bin_size: float = 0.01,
) -> pd.DataFrame:
    """Bin a hawk DataFrame, casting hawk-specific columns to float.

    Wraps `bin_dataframe_means`, automatically
    casting ``HorzDistance`` and ``VertDistance`` to float before
    binning.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Hawk flight DataFrame (e.g. from `load_bird_data`).
    x_axis : str
        Column to bin along, typically ``'HorzDistance'`` (spatial) or
        ``'time'`` (temporal).
    bin_size : float
        Width of each bin in the units of *x_axis* (metres for
        distance, seconds for time).

    Returns
    -------
    pd.DataFrame
        Binned DataFrame with one row per bin, containing group means
        of all numeric columns.
    """
    return bin_dataframe_means(
        dataframe,
        x_axis,
        bin_size,
        numeric_cast_columns=["HorzDistance", "VertDistance"],
    )


# ── DMD wrappers ────────────────────────────────────────────────────


def run_hawk_dmd(
    data: np.ndarray,
    times: np.ndarray | None = None,
    n_modes: int = 6,
    d: int = 2,
    eig_constraints: set[str] | None = None,
    n_markers: int = 8,
    mean_shape_path: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> DMDResult:
    """Run DMD with hawk defaults.

    Loads the mean hawk shape automatically and delegates to
    `run_dmd`.  This is the recommended entry
    point for single-array hawk analyses where you already have the
    marker data in memory.

    Parameters
    ----------
    data : np.ndarray
        Marker positions, shape ``(n_frames, n_markers, 3)`` or
        ``(n_frames, n_markers * 3)``.
    times : np.ndarray, optional
        Time vector, shape ``(n_frames,)``.  If ``None``, a uniform
        time step is assumed.
    n_modes : int
        Number of DMD modes (SVD rank).
    d : int
        Hankel delay-embedding depth.
    eig_constraints : set of str, optional
        Constraints passed to BOPDMD, e.g. ``{"conjugate_pairs"}``.
        Defaults to ``{"conjugate_pairs"}``.
    n_markers : int
        Number of anatomical markers (8 bilateral, 4 unilateral).
    mean_shape_path : str, optional
        Path to the mean-shape CSV.  Defaults to the bundled file.
    verbose : bool
        If ``True``, print fitting diagnostics.
    **kwargs
        Additional keyword arguments forwarded to `run_dmd`
        (e.g. *init_alpha*).

    Returns
    -------
    DMDResult
        Complete DMD results including eigenvalues, modes,
        reconstruction, etc.
    """
    if eig_constraints is None:
        eig_constraints = {"conjugate_pairs"}
    avg = get_average_shape(n_markers, mean_shape_path)
    return run_dmd(
        data=data,
        times=times,
        n_modes=n_modes,
        d=d,
        eig_constraints=eig_constraints,
        average_shape=avg,
        n_markers=n_markers,
        verbose=verbose,
        **kwargs,
    )


def run_sequence_dmd(
    bird_name: str,
    perch_dist: str,
    turn: str,
    behaviour: str,
    seqID: str,
    n_modes: int = 10,
    d: int = 2,
    eig_constraints: set[str] | None = None,
    min_seq_length: int | None = None,
    interpolate: bool = False,
    verbose: bool = True,
) -> DMDResult | None:
    """Load one hawk sequence from disk and run DMD on it.

    Convenience wrapper that chains `load_bird_data`,
    `load_sequence_data`, optional spline
    interpolation, and `run_dmd` into a single
    call.

    Parameters
    ----------
    bird_name : str
        Individual hawk identifier, e.g. ``'Toothless'``.
    perch_dist : str
        Distance to perch, e.g. ``'9m'``.
    turn : str
        Turn direction: ``'Straight'``, ``'Left'``, or ``'Right'``.
    behaviour : str
        Flight phase label, e.g. ``'Initial'``, ``'Flapping'``.
    seqID : str
        Unique sequence identifier within the loaded dataset.
    n_modes : int
        Number of DMD modes (SVD rank).
    d : int
        Hankel delay-embedding depth.
    eig_constraints : set of str, optional
        Constraints passed to BOPDMD, e.g. ``{"conjugate_pairs"}``.
        Defaults to ``{"conjugate_pairs"}``.
    min_seq_length : int, optional
        Minimum number of frames required.  Sequences shorter than
        this are skipped.  Defaults to ``n_modes + 1``.
    interpolate : bool
        If ``True``, apply cubic-spline interpolation to produce
        uniformly spaced time steps before fitting.
    verbose : bool
        If ``True``, print loading and fitting diagnostics.

    Returns
    -------
    DMDResult or None
        Complete DMD results, or ``None`` if the sequence has fewer
        frames than *min_seq_length*.
    """
    if eig_constraints is None:
        eig_constraints = {"conjugate_pairs"}
    df, marker_cols = load_bird_data(
        bird_name=bird_name,
        behaviour=behaviour,
        perch_distance=perch_dist,
        turn=turn,
        verbose=verbose,
    )
    df = remove_time_duplicates(df)

    n_markers = len(marker_cols) // 3
    average_shape = get_average_shape(n_markers)
    markers, times = load_sequence_data(df, seqID, marker_cols)

    if min_seq_length is None:
        min_seq_length = n_modes + 1
    if markers.shape[0] <= min_seq_length:
        if verbose:
            print(f"Sequence {seqID} too short ({markers.shape[0]} frames)")
        return None

    if interpolate:
        times, markers = spline_interpolation(times, markers)

    markers = markers.reshape(-1, n_markers, 3)
    return run_dmd(
        data=markers,
        times=times,
        n_modes=n_modes,
        d=d,
        eig_constraints=eig_constraints,
        average_shape=average_shape,
        n_markers=n_markers,
        verbose=verbose,
    )


# ── Batch analysis ──────────────────────────────────────────────────


def batch_rmse_analysis(
    df: pd.DataFrame,
    seqIDs: list[str],
    marker_column_names: np.ndarray,
    average_shape: np.ndarray,
    n_modes: int = 6,
    d: int = 2,
    marker_names: list[str] | None = None,
    verbose: bool = False,
    per_frame: bool = False,
) -> pd.DataFrame:
    """Run DMD on many hawk sequences and collect RMSE statistics.

    Iterates over *seqIDs*, fits a DMD model to each sequence, and
    computes the root-mean-square reconstruction error.  Sequences
    that fail (e.g. due to convergence issues) are silently skipped
    unless *verbose* is ``True``.

    Parameters
    ----------
    df : pd.DataFrame
        Full hawk dataset containing all sequences (e.g. from
        `load_bird_data`).
    seqIDs : list of str
        Sequence identifiers to analyse.
    marker_column_names : np.ndarray
        1-D string array of marker column names, length
        ``n_markers * 3``.
    average_shape : np.ndarray
        Mean hawk shape, shape ``(1, n_markers, 3)``, subtracted
        before fitting.
    n_modes : int
        Number of DMD modes (SVD rank).
    d : int
        Hankel delay-embedding depth.
    marker_names : list of str, optional
        Human-readable marker names used as suffixes for per-marker
        RMSE columns (e.g. ``rmse_wingtip``).  Defaults to
        ``marker_0``, ``marker_1``, etc.
    verbose : bool
        If ``True``, print a message for each skipped sequence.
    per_frame : bool
        If ``True``, each result row also contains:

        - ``rmse_per_frame`` — 1-D array of frame-level RMSE values.
        - ``rmse_per_frame_per_marker`` — 2-D array, shape
          ``(n_frames, n_markers)``.
        - ``horz_distance`` — 1-D array of horizontal distance per
          frame (only if the column exists in *df*).

    Returns
    -------
    pd.DataFrame
        One row per successfully analysed sequence, with columns
        ``seqID``, ``total_rmse``, per-marker RMSE columns, and any
        available metadata from ``HAWK_META_COLUMNS``.
    """
    n_markers = len(marker_column_names) // 3
    meta_cols = [c for c in HAWK_META_COLUMNS if c in df.columns]
    has_horz = per_frame and "HorzDistance" in df.columns
    rows = []

    for sid in seqIDs:
        try:
            seq_df = df[df["seqID"] == sid]
            metadata = seq_df.iloc[0][meta_cols].to_dict() if meta_cols else {}

            markers, times = load_sequence_data(df, sid, marker_column_names)
            markers = markers.reshape(-1, n_markers, 3)
            times_unique, unique_idx = np.unique(times, return_index=True)
            markers = markers[unique_idx]
            times = times_unique

            result = run_dmd(
                data=markers,
                times=times,
                n_modes=n_modes,
                d=d,
                eig_constraints={"conjugate_pairs"},
                n_markers=n_markers,
                average_shape=average_shape,
                verbose=False,
            )

            gt = markers[:-1]
            sq_err = (gt - result.reconstruction) ** 2
            total_rmse = float(np.sqrt(np.mean(sq_err)))
            rmse_per_marker = np.sqrt(np.mean(sq_err, axis=(0, 2)))

            row = {"seqID": sid, "total_rmse": total_rmse, **metadata}
            names = marker_names or [f"marker_{i}" for i in range(n_markers)]
            for i, name in enumerate(names):
                row[f"rmse_{name}"] = rmse_per_marker[i]

            if per_frame:
                row["rmse_per_frame"] = np.sqrt(np.mean(sq_err, axis=(1, 2)))
                row["rmse_per_frame_per_marker"] = np.sqrt(
                    np.mean(sq_err, axis=2)
                )  # shape (n_frames, n_markers)
            if has_horz:
                hd = seq_df["HorzDistance"].to_numpy(dtype=np.float64)
                row["horz_distance"] = hd[unique_idx][:-1]

            rows.append(row)
        except Exception as exc:
            if verbose:
                print(f"Skipping {sid}: {exc}")

    return pd.DataFrame(rows)


# ── Hawk plots ──────────────────────────────────────────────────────


def plot_hawk_markers(
    dataframe: pd.DataFrame,
    _marker_column_names: np.ndarray,
    x_axis: str = "HorzDistance",
) -> Figure:
    """Plot a 4x3 scatter grid of marker positions.

    Each row corresponds to one of the four marker positions (see
    ``MARKER_POSITIONS``); each column shows the x, y, or z
    coordinate.  Left and right markers are overlaid in the same
    colour.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Hawk flight DataFrame with marker columns following the
        ``{side}_{position}_{axis}`` naming convention.
    _marker_column_names : np.ndarray
        1-D string array of marker column names (used for layout
        consistency; individual columns are constructed from
        ``MARKER_POSITIONS``).
    x_axis : str
        Column to plot on the horizontal axis.  ``'HorzDistance'``
        (negated automatically) or ``'time'``.

    Returns
    -------
    Figure
        Matplotlib figure with 12 scatter subplots.
    """
    fig, axes = plt.subplots(4, 3, figsize=(6, 8), sharex=True, sharey=False)
    ax = axes.flatten()
    coordinates = ["x", "y", "z"]

    for pi, pos in enumerate(MARKER_POSITIONS):
        for ci, coord in enumerate(coordinates):
            idx = pi * 3 + ci
            left = f"left_{pos}_{coord}"
            right = f"right_{pos}_{coord}"
            negate = x_axis.lower().startswith("horz")
            x = -dataframe[x_axis] if negate else dataframe[x_axis]

            ax[idx].scatter(x, dataframe[left], s=2, color=MARKER_COLOURS[pos])
            ax[idx].scatter(x, dataframe[right], s=2, color=MARKER_COLOURS[pos])

            ax[idx].grid(
                True,
                which="major",
                axis="x",
                linestyle="-",
                linewidth=0.5,
                color="k",
                alpha=0.5,
            )
            ax[idx].xaxis.set_major_locator(MaxNLocator(4))

            if ci == 0:
                ax[idx].set_ylabel(pos, fontsize=10)
            if pi == len(MARKER_POSITIONS) - 1:
                ax[idx].set_xlabel(x_axis, fontsize=10)
            if pi == 0:
                ax[idx].set_title(coord, fontsize=12)

            ax[idx].yaxis.set_ticklabels([])
            _remove_spines(ax[idx])

    plt.tight_layout()
    return fig


def plot_hawk_markers_shaded(
    dataframe: pd.DataFrame,
    _marker_column_names: np.ndarray,
    x_axis: str = "HorzDistance",
    bin_size: float = 0.01,
) -> Figure:
    """Plot a 4x3 grid of binned mean +/- 1 SD bands for hawk markers.

    Same layout as `plot_hawk_markers` but instead of raw
    scatter points, data are binned along *x_axis* and the mean is
    shown as a solid (left) or dotted (right) line with a shaded
    +/- 1 standard deviation band.  The x-coordinate uses absolute
    values before averaging.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Hawk flight DataFrame with marker columns following the
        ``{side}_{position}_{axis}`` naming convention.
    _marker_column_names : np.ndarray
        1-D string array of marker column names.
    x_axis : str
        Column to bin along.  ``'HorzDistance'`` (negated
        automatically) or ``'time'``.
    bin_size : float
        Width of each bin in the units of *x_axis* (metres for
        distance, seconds for time).

    Returns
    -------
    Figure
        Matplotlib figure with 12 shaded-band subplots.
    """
    fig, axes = plt.subplots(4, 3, figsize=(6, 8), sharex=True, sharey=False)
    ax = axes.flatten()
    coordinates = ["x", "y", "z"]

    x_min = dataframe[x_axis].min()
    x_max = dataframe[x_axis].max()
    bins = np.arange(x_min, x_max + bin_size, bin_size)
    bins = np.around(bins, 3)
    bin_centres = bins[:-1] + bin_size / 2

    df = dataframe.copy()
    df["bins"] = pd.cut(df[x_axis], bins.tolist(), right=False, include_lowest=True)

    for pi, pos in enumerate(MARKER_POSITIONS):
        for ci, coord in enumerate(coordinates):
            idx = pi * 3 + ci
            left = f"left_{pos}_{coord}"
            right = f"right_{pos}_{coord}"
            colour = MARKER_COLOURS[pos]
            negate = x_axis.lower().startswith("horz")

            if coord == "x":
                lm = df.groupby("bins", observed=True)[left].apply(
                    lambda s: np.abs(s).mean()
                )
                ls = df.groupby("bins", observed=True)[left].apply(
                    lambda s: np.abs(s).std()
                )
                rm = df.groupby("bins", observed=True)[right].apply(
                    lambda s: np.abs(s).mean()
                )
                rs = df.groupby("bins", observed=True)[right].apply(
                    lambda s: np.abs(s).std()
                )
            else:
                lm = df.groupby("bins", observed=True)[left].mean()
                ls = df.groupby("bins", observed=True)[left].std()
                rm = df.groupby("bins", observed=True)[right].mean()
                rs = df.groupby("bins", observed=True)[right].std()

            vl = ~np.isnan(lm) & ~np.isnan(ls)
            vr = ~np.isnan(rm) & ~np.isnan(rs)
            lbc = bin_centres[: len(lm)][vl]
            rbc = bin_centres[: len(rm)][vr]

            if np.any(vl):
                xv = -lbc if negate else lbc
                ax[idx].fill_between(
                    xv,
                    lm[vl] - ls[vl],
                    lm[vl] + ls[vl],
                    color=colour,
                    alpha=0.3,
                    edgecolor="none",
                )
                ax[idx].plot(
                    xv,
                    lm[vl],
                    color=colour,
                    linewidth=2,
                    linestyle="-",
                    label="Left" if idx == 0 else None,
                )

            if np.any(vr):
                xv = -rbc if negate else rbc
                ax[idx].fill_between(
                    xv,
                    rm[vr] - rs[vr],
                    rm[vr] + rs[vr],
                    color=colour,
                    alpha=0.3,
                    edgecolor="none",
                )
                ax[idx].plot(
                    xv,
                    rm[vr],
                    color=colour,
                    linewidth=2,
                    linestyle=":",
                    label="Right" if idx == 0 else None,
                )

            if ci == 0:
                ax[idx].set_ylabel(pos, fontsize=10)

            titles = ["abs(x)", "y", "z"]
            if pi == 0:
                ax[idx].set_title(titles[ci], fontsize=12)

            ax[idx].yaxis.set_ticklabels([])
            _remove_spines(ax[idx])

            if pi == len(MARKER_POSITIONS) - 1 and ci == 0:
                _format_bottom_axis(ax[idx], dataframe, x_axis, negate)

    plt.tight_layout()
    return fig


def _format_bottom_axis(ax, dataframe, x_axis: str, negate: bool) -> None:
    """Add x-axis labels and tick marks to the bottom-left subplot."""
    x_label = "Horizontal Distance to perch (m)" if negate else "time (s)"
    ax.set_xlabel(x_label, fontsize=10)
    xr = round(dataframe[x_axis].max(), 1)
    ax.set_xticks([0, xr])
    ax.set_xticklabels([0, xr])
    y_lo, y_hi = ax.get_ylim()
    ax.hlines(y=y_lo, xmin=0, xmax=xr, color="k", linewidth=2)
    tick = abs(y_hi) * 0.04
    ax.vlines(x=0, ymin=y_lo, ymax=y_lo + tick, color="k", linewidth=1.5)
    ax.vlines(x=xr, ymin=y_lo, ymax=y_lo + tick, color="k", linewidth=1.5)


def _remove_spines(ax, keep_bottom: bool = False):
    """Strip spines and ticks from an Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify.
    keep_bottom : bool
        If ``True``, preserve the bottom spine.
    """
    for spine in ("top", "right", "left", "bottom"):
        if spine == "bottom" and keep_bottom:
            continue
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)


def plot_hawk_2d(
    dataframe: pd.DataFrame,
    marker_column_names: np.ndarray,
) -> Figure:
    """Plot 2x4 y-z projections of hawk marker positions.

    Each subplot shows one marker's lateral (y) vs. vertical (z)
    position across all frames, coloured by time.  Useful for
    visualising the spatial envelope of wing kinematics.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Hawk flight DataFrame with marker columns and a ``time``
        column.
    marker_column_names : np.ndarray
        1-D string array of marker column names, length
        ``n_markers * 3``.  Every third name (stride 3) is used to
        derive the marker base names.

    Returns
    -------
    Figure
        Matplotlib figure with 8 equal-aspect subplots.
    """
    names = marker_column_names[::3]
    names = [m.split("_x")[0] for m in names]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(5, 8),
        sharex=True,
        sharey=True,
        tight_layout=True,
    )
    ax = axes.flatten()

    for ii, marker in enumerate(names):
        ax[ii].scatter(
            dataframe[marker + "_y"],
            dataframe[marker + "_z"],
            s=0.1,
            c=dataframe["time"],
        )
        ax[ii].set_aspect("equal", "box")
        ax[ii].axis("off")
        ax[ii].set_title(marker, fontsize=8)

    return fig


def plot_single_sequence(
    dataframe: pd.DataFrame,
    seq_num: int,
    marker_name: str = "right_wingtip_z",
    x_axis: str = "HorzDistance",
) -> Figure:
    """Plot a single hawk sequence for one marker coordinate.

    Selects the sequence at position *seq_num* (zero-indexed) from the
    unique ``seqID`` values and scatter-plots the chosen marker
    coordinate against the horizontal axis, coloured by time.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Hawk flight DataFrame with ``seqID``, ``time``, marker, and
        *x_axis* columns.
    seq_num : int
        Zero-based index into the sorted unique sequence IDs.
    marker_name : str
        Column name of the marker coordinate to plot on the vertical
        axis, e.g. ``'right_wingtip_z'``.
    x_axis : str
        Column for the horizontal axis.  ``'HorzDistance'`` labels the
        axis as distance to perch; ``'time'`` labels it as time from
        take-off.

    Returns
    -------
    Figure
        Matplotlib figure with a single scatter subplot.
    """
    seqIDs = dataframe["seqID"].unique()
    seqID = seqIDs[seq_num]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    mask = dataframe["seqID"] == seqID
    ax.scatter(
        dataframe[mask][x_axis],
        dataframe[mask][marker_name],
        s=10,
        c=dataframe[mask]["time"],
    )
    if x_axis.startswith("Horz"):
        ax.set_xlabel("Horizontal Distance to Perch (m)")
    elif x_axis.startswith("time"):
        ax.set_xlabel("time from take-off jump (s)")
    else:
        ax.set_xlabel(x_axis)
    ax.set_ylabel(marker_name)
    return fig
