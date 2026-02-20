# ruff: noqa: RUF001, RUF002
"""Core DMD analysis: fitting, result packaging, and convergence.

This module contains the central ``run_dmd`` entry point, the
``DMDResult`` dataclass that replaces all tuple returns, conjugate-pair
detection, and mode-convergence analysis.

Functions
---------
run_dmd
    Validate, normalise, fit BOPDMD, reorder, reconstruct, and return
    a ``DMDResult``.
detect_conjugate_pairs
    Identify complex-conjugate eigenvalue pairs.
convergence_analysis
    Sweep mode counts and collect RMSE / variance explained.

Classes
-------
DMDResult
    Frozen dataclass holding every output of a DMD analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydmd.bopdmd import BOPDMD
from pydmd.preprocessing.hankel import hankel_preprocessing

from ._constants import (
    CONJUGATE_TOLERANCE,
    DEFAULT_DELAY,
    DEFAULT_EIG_CONSTRAINTS,
    DEFAULT_N_MODES,
)
from .data import normalise_data, validate_marker_data
from .stats import compute_rmse, variance_explained

# ── DMDResult dataclass ─────────────────────────────────────────────


@dataclass(frozen=True)
class DMDResult:
    """Complete results from a DMD analysis.

    Notation
    --------
    The field names use plain English for readability.  The standard
    mathematical symbols from DMD literature (Kutz *et al.*, Tu
    *et al.*) are listed here for cross-reference:

        Field               Math symbol     Equation
        ─────────────────   ─────────────   ────────────────────────
        eigenvalues         λ = σ + iω      x(t) = Σ bⱼ φⱼ exp(λⱼt)
        modes               φ (Phi)         spatial DMD modes
        amplitudes          b               mode weights / coefficients
        frequencies_hz      Im(λ) / 2π      oscillation frequency
        growth_rates        σ = Re(λ)       exponential growth/decay
        mode_magnitudes     |φ|             element-wise magnitude
        phase_shifts        θ               arctan2(-sign(ω)·Im(φ), Re(φ))
    """

    eigenvalues: np.ndarray
    """λ = σ + iω — complex eigenvalues, shape ``(n_modes,)``."""

    modes: np.ndarray
    """φ (Phi) — complex spatial modes, shape ``(n_coords, n_modes)``."""

    amplitudes: np.ndarray
    """b — complex mode weights, shape ``(n_modes,)``."""

    frequencies_hz: np.ndarray
    """Im(λ) / (2π) — oscillation frequency in Hz, shape ``(n_modes,)``."""

    growth_rates: np.ndarray
    """σ = Re(λ) — exponential growth/decay rate, shape ``(n_modes,)``."""

    mode_magnitudes: np.ndarray
    """|φ| — magnitude of spatial modes, shape ``(n_markers, 3, n_modes)``."""

    phase_shifts: np.ndarray
    """θ — phase angles, same shape as ``modes``."""

    conjugate_pairs: list[tuple[int, int]]
    """Paired eigenvalue indices ``[(i, j), ...]``."""

    reconstruction: np.ndarray
    """Full reconstruction with mean shape, ``(n_frames, n_markers, 3)``."""

    average_shape: np.ndarray
    """Mean shape subtracted before fitting."""

    times: np.ndarray
    """Time vector used for fitting."""

    sort_indices: np.ndarray
    """Maps amplitude-sorted position to original BOPDMD index."""

    _dmd_object: Any
    """Underlying ``BOPDMD`` object — for advanced use only."""

    # ── Helper properties ───────────────────────────────────────────

    @property
    def n_modes(self) -> int:
        """Number of DMD modes."""
        return len(self.eigenvalues)

    @property
    def n_pairs(self) -> int:
        """Number of conjugate pairs."""
        return len(self.conjugate_pairs)

    def pair_frequency(self, pair_index: int) -> float:
        """Frequency (Hz) of the *pair_index*-th conjugate pair.

        Parameters
        ----------
        pair_index : int
            Zero-based pair index.

        Returns
        -------
        float
            Absolute frequency in Hz.

        Notes
        -----
        Uses the original (unsorted) eigenvalues from the BOPDMD
        object, since ``conjugate_pairs`` indices are in that space.
        """
        i, j = self.conjugate_pairs[pair_index]
        # eigenvalues are stored unsorted in DMDResult
        # but conjugate_pairs indices are in the unsorted space too
        # So we need the original eigenvalues:
        dmd_eigs = self._dmd_object.eigs
        return float(np.mean(np.abs(np.imag(dmd_eigs[[i, j]]))) / (2 * np.pi))

    def pair_indices(self, pair_numbers: list[int]) -> list[int]:
        """Return flat mode indices for the given pair numbers.

        Parameters
        ----------
        pair_numbers : list of int
            Zero-based pair numbers.

        Returns
        -------
        list of int
            Mode indices belonging to the selected pairs.

        Example
        -------
        >>> result.pair_indices([0, 2])
        [0, 1, 4, 5]
        """
        indices = []
        for pair in pair_numbers:
            if pair < 0 or pair >= len(self.conjugate_pairs):
                msg = f"Pair {pair} out of range [0, {len(self.conjugate_pairs) - 1}]"
                raise ValueError(msg)
            ii, jj = self.conjugate_pairs[pair]
            indices.append(ii)
            if ii != jj:
                indices.append(jj)
        return indices


# ── Conjugate-pair detection ────────────────────────────────────────


def detect_conjugate_pairs(
    eigenvalues: np.ndarray,
    tolerance: float = CONJUGATE_TOLERANCE,
) -> list[tuple[int, int]]:
    """Identify complex-conjugate eigenvalue pairs.

    Two eigenvalues λ_i and λ_j are conjugates when their real parts
    match and their imaginary parts sum to zero (within *tolerance*).
    Unmatched eigenvalues are self-paired ``(i, i)`` and treated as
    real modes.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Complex eigenvalues, shape ``(n_modes,)``.
    tolerance : float
        Matching tolerance for real and imaginary comparisons.

    Returns
    -------
    list of (int, int)
        Pairs in the order they appear in *eigenvalues*.
    """
    n = len(eigenvalues)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for i in range(n):
        if i in used:
            continue
        matched = False
        for j in range(i + 1, n):
            if j in used:
                continue
            if (
                abs(eigenvalues[i].real - eigenvalues[j].real) < tolerance
                and abs(eigenvalues[i].imag + eigenvalues[j].imag) < tolerance
            ):
                pairs.append((i, j))
                used.update((i, j))
                matched = True
                break
        if not matched:
            pairs.append((i, i))
            used.add(i)
    return pairs


# ── Internal helpers ────────────────────────────────────────────────


def _fit_bopdmd(
    data: np.ndarray,
    times: np.ndarray,
    n_modes: int,
    eig_constraints: set[str],
    d: int,
    init_alpha: np.ndarray | None = None,
) -> Any:
    """Fit a Hankel-preprocessed BOPDMD model.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_coords, n_frames)`` — transposed and normalised.
    times : np.ndarray
        Shape ``(n_frames,)``.
    n_modes : int
        SVD rank (number of modes).
    eig_constraints : set of str
        Passed to ``BOPDMD``.
    d : int
        Hankel delay embedding depth.
    init_alpha : np.ndarray, optional
        Initial eigenvalue guesses.

    Returns
    -------
    Any
        Fitted Hankel-preprocessed BOPDMD model.

    Raises
    ------
    RuntimeError
        If the BOPDMD fit fails.
    """
    kwargs: dict[str, Any] = {"svd_rank": n_modes, "eig_constraints": eig_constraints}
    if init_alpha is not None:
        kwargs["init_alpha"] = init_alpha

    dmd: Any = hankel_preprocessing(BOPDMD(**kwargs), d=d)

    dt = np.diff(times)
    if np.any(dt <= 0):
        msg = "Time vector must be strictly increasing"
        raise ValueError(msg)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Initial trial of Optimized DMD failed to converge",
                category=UserWarning,
            )
            dmd.fit(data, t=times[1:])
    except Exception as exc:
        msg = f"BOPDMD fit failed: {exc}"
        raise RuntimeError(msg) from exc

    return dmd


def _reorder_by_amplitude(dmd: Any, n_markers: int, n_spatial_dims: int = 3):
    """Sort BOPDMD results by descending amplitude magnitude.

    Returns
    -------
    frequencies_hz, mode_magnitudes, amplitudes, modes, phase_shifts, sort_idx
    """
    frequencies = np.imag(dmd.eigs)
    sign_omega = np.sign(frequencies)
    sort_idx = np.argsort(np.abs(dmd.amplitudes))[::-1]

    n_coords = n_markers * n_spatial_dims
    modes = dmd.modes[:n_coords, sort_idx]
    amplitudes = dmd.amplitudes[sort_idx]
    frequencies_sorted = frequencies[sort_idx]

    # Mode magnitudes  — |φ|
    magnitudes = np.abs(modes)
    mode_magnitudes = magnitudes.reshape(n_markers, n_spatial_dims, -1)

    # Phase shifts — θ = arctan2(-sign(ω) · Im(φ), Re(φ))
    # phase = arctan2(-sign(ω) · Im(φ), Re(φ))  [phasor notation]
    phase_shifts = np.arctan2(-sign_omega[sort_idx] * np.imag(modes), np.real(modes))

    frequencies_hz = frequencies_sorted / (2 * np.pi)

    return frequencies_hz, mode_magnitudes, amplitudes, modes, phase_shifts, sort_idx


def _reconstruct_complex(
    times: np.ndarray,
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Standard DMD reconstruction using full complex eigenvalues.

    Computes:

        x̃(t) = Re[ Σ_i  b_i · φ_i · exp(λ_i · t) ]

    Parameters
    ----------
    times : np.ndarray
        Time vector, shape ``(n_times,)``.
    eigenvalues : np.ndarray
        Complex eigenvalues, shape ``(n_modes,)``.
    modes : np.ndarray
        Complex spatial modes, shape ``(n_coords, n_modes)``.
    amplitudes : np.ndarray
        Complex amplitudes, shape ``(n_modes,)``.

    Returns
    -------
    np.ndarray
        Real-valued reconstruction, shape ``(n_times, n_coords)``.
    """
    # Vectorised: (n_modes, n_times) = exp(λ · t)
    dynamics = np.exp(np.outer(eigenvalues, times))  # (n_modes, n_times)
    # (n_coords, n_times) = modes @ diag(amplitudes) @ dynamics
    weighted = modes * amplitudes[np.newaxis, :]  # (n_coords, n_modes)
    result = weighted @ dynamics  # (n_coords, n_times)
    return np.real(result.T)  # (n_times, n_coords)


# ── Public entry point ──────────────────────────────────────────────


def run_dmd(
    data: np.ndarray,
    times: np.ndarray | None = None,
    n_modes: int = DEFAULT_N_MODES,
    d: int = DEFAULT_DELAY,
    eig_constraints: set[str] = DEFAULT_EIG_CONSTRAINTS,
    average_shape: np.ndarray | None = None,
    n_markers: int | None = None,
    init_alpha: np.ndarray | None = None,
    verbose: bool = True,
) -> DMDResult:
    """Run a complete DMD analysis on marker data.

    This is the main entry point.  It validates the input, centres the
    data, fits a Hankel-preprocessed BOPDMD model, reorders results by
    amplitude, reconstructs the full time series, and packages
    everything into a `DMDResult`.

    Parameters
    ----------
    data : np.ndarray
        Marker data — shape ``(n_frames, n_markers, 3)`` or
        ``(n_frames, n_markers*3)``.
    times : np.ndarray
        Timestamps, shape ``(n_frames,)``.
    n_modes : int
        Number of DMD modes (SVD rank).
    d : int
        Hankel delay embedding depth.
    eig_constraints : set of str
        Eigenvalue constraints for BOPDMD.
    average_shape : np.ndarray, optional
        Mean shape to subtract before fitting.  Required for marker
        data; omit only for pre-centred data.
    n_markers : int, optional
        Number of markers.  Required when *average_shape* is provided.
    init_alpha : np.ndarray, optional
        Initial eigenvalue guesses for the BOPDMD optimiser.
    verbose : bool
        Print progress messages.

    Returns
    -------
    DMDResult
        All outputs packaged in a frozen dataclass.

    Raises
    ------
    ValueError
        If input shapes are invalid or *average_shape* / *n_markers*
        are missing when required.
    RuntimeError
        If the BOPDMD fit fails.

    Examples
    --------
    >>> result = run_dmd(markers, times, n_modes=6, d=2, average_shape=avg, n_markers=8)
    >>> result.frequencies_hz
    array([ 4.51, -4.51,  8.76, -8.76, -0.00,  0.00])
    """
    # Validate and flatten
    data_flat, n_frames, detected_markers, n_coords = validate_marker_data(data)

    if n_markers is None:
        n_markers = detected_markers

    if times is None:
        times = np.arange(n_frames, dtype=float)
    if len(times) != n_frames:
        msg = f"Time vector length ({len(times)}) != frame count ({n_frames})"
        raise ValueError(msg)

    if average_shape is None:
        msg = "average_shape must be provided for marker data"
        raise ValueError(msg)

    avg_flat = average_shape.reshape(1, -1)
    if avg_flat.shape[1] != n_coords:
        msg = f"average_shape has {avg_flat.shape[1]} coords, data has {n_coords}"
        raise ValueError(msg)

    if verbose:
        print(f"Running DMD with {n_modes} modes, delay d={d}")
        print(f"Input shape: {data_flat.shape}")

    # Centre and transpose for BOPDMD: (n_coords, n_frames)
    centred = normalise_data(data_flat, avg_flat)
    data_T = centred.T if centred.shape[0] > centred.shape[1] else centred
    if data_T.shape[0] > data_T.shape[1]:
        data_T = data_T.T

    # Fit
    dmd = _fit_bopdmd(data_T, times, n_modes, eig_constraints, d, init_alpha)

    if verbose:
        print(f"Number of variables in DMD results: {dmd.modes.shape[0]}")
        print(f"Complex Eigenvalues (log): {np.round(dmd.eigs, 3)}")
        print(f"Number of modes in Psi: {dmd.modes.shape[1]}")

    # Reorder
    freq_hz, mag, amps, modes, phase, sort_idx = _reorder_by_amplitude(dmd, n_markers)

    # Reconstruct using unsorted eigenvalues (matches old behaviour)
    recon_flat = _reconstruct_complex(
        times[1:], dmd.eigs, dmd.modes[:n_coords, :], dmd.amplitudes
    )
    recon = recon_flat.reshape(-1, n_markers, 3) + average_shape.reshape(
        -1, n_markers, 3
    )

    # Conjugate pairs
    pairs = detect_conjugate_pairs(dmd.eigs)

    return DMDResult(
        eigenvalues=dmd.eigs,
        modes=modes,
        amplitudes=amps,
        frequencies_hz=freq_hz,
        growth_rates=np.real(dmd.eigs),
        mode_magnitudes=mag,
        phase_shifts=phase,
        conjugate_pairs=pairs,
        reconstruction=recon,
        average_shape=average_shape,
        times=times,
        sort_indices=sort_idx,
        _dmd_object=dmd,
    )


# ── Convergence analysis ────────────────────────────────────────────


def convergence_analysis(
    markers: np.ndarray,
    times: np.ndarray,
    average_shape: np.ndarray,
    n_markers: int,
    max_modes: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run DMD with increasing mode counts and track accuracy.

    Parameters
    ----------
    markers : np.ndarray
        Shape ``(n_frames, n_markers, 3)``.
    times : np.ndarray
        Time vector.
    average_shape : np.ndarray
        Mean shape.
    n_markers : int
        Number of markers.
    max_modes : int, optional
        Maximum modes to test.  Defaults to
        ``min(20, n_frames // 2)``.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: ``n_modes``, ``rmse_mean``, ``rmse_std``,
        ``variance_explained``, ``dmd_results``, ``reconstructions``.
    """
    if max_modes is None:
        max_modes = min(20, markers.shape[0] // 2)

    results: dict[str, list] = {
        "n_modes": [],
        "rmse_mean": [],
        "rmse_std": [],
        "variance_explained": [],
        "dmd_results": [],
        "reconstructions": [],
    }

    for nm in range(2, max_modes + 1, 2):
        try:
            res = run_dmd(
                data=markers,
                times=times,
                n_modes=nm,
                d=DEFAULT_DELAY,
                eig_constraints=DEFAULT_EIG_CONSTRAINTS,
                average_shape=average_shape,
                n_markers=n_markers,
                verbose=False,
            )
        except RuntimeError:
            if verbose:
                print(f"  DMD failed for {nm} modes")
            continue

        ground = markers[1:]
        rmse = compute_rmse(res.reconstruction, ground)

        results["n_modes"].append(nm)
        results["rmse_mean"].append(float(np.mean(rmse)))
        results["rmse_std"].append(float(np.std(rmse)))
        results["variance_explained"].append(
            variance_explained(ground, res.reconstruction)
        )
        results["dmd_results"].append(res)
        results["reconstructions"].append(res.reconstruction)

        if verbose:
            print(
                f"  {nm:2d} modes — RMSE {np.mean(rmse):.6f} ± "
                f"{np.std(rmse):.6f}, VE {results['variance_explained'][-1]:.4f}"
            )

    return results
