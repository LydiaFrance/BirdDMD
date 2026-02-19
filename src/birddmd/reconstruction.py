"""Reconstruction and forecasting from DMD results.

Two functions replace the seven in the old ``dmd_reconstruction`` module:

* :func:`reconstruct` — rebuild the signal from selected modes/pairs,
  with optional stabilisation.
* :func:`forecast` — like ``reconstruct`` but designed for extrapolation,
  with eigenvalue modification support.

Functions
---------
reconstruct
    Rebuild the signal from a DMDResult using selected modes or pairs.
forecast
    Extrapolate beyond the training window, optionally modifying
    eigenvalues for stabilisation or frequency changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._constants import ZERO_FREQUENCY_THRESHOLD

if TYPE_CHECKING:
    from .core import DMDResult


# ── Internal reconstruction engines ─────────────────────────────────


def _reconstruct_complex(
    times: np.ndarray,
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    amplitudes: np.ndarray,
    mode_indices: list[int] | None = None,
) -> np.ndarray:
    """Standard DMD reconstruction using full complex eigenvalues.

    Computes:

        x̃(t) = Re[ Σ_i  b_i · φ_i · exp(λ_i · t) ]

    Parameters
    ----------
    times : np.ndarray
        Shape ``(n_times,)``.
    eigenvalues, modes, amplitudes
        From the underlying BOPDMD object.
    mode_indices : list of int, optional
        Subset of mode indices to include.  ``None`` means all.

    Returns
    -------
    np.ndarray
        Real-valued, shape ``(n_times, n_coords)``.
    """
    if mode_indices is not None:
        eigenvalues = eigenvalues[mode_indices]
        modes = modes[:, mode_indices]
        amplitudes = amplitudes[mode_indices]

    dynamics = np.exp(np.outer(eigenvalues, times))
    weighted = modes * amplitudes[np.newaxis, :]
    return np.real((weighted @ dynamics).T)


def _reconstruct_phasor(
    times: np.ndarray,
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    amplitudes: np.ndarray,
    conjugate_pairs: list,
    mode_indices: list[int] | None = None,
) -> np.ndarray:
    """Phasor (stable) reconstruction using only frequencies.

    Computes:

        x̃(t) = Σ_{pairs}  factor · |b| · |φ| · cos(|ω|t - θ)

    where ``factor = 2`` for true conjugate pairs and ``1`` for
    real (self-paired) modes.  Growth/decay is stripped because only
    the imaginary part of each eigenvalue is used.

    Parameters
    ----------
    times : np.ndarray
        Shape ``(n_times,)``.
    eigenvalues : np.ndarray
        Full complex eigenvalues (imaginary parts are extracted).
    modes : np.ndarray
        Complex spatial modes, shape ``(n_coords, n_modes)``.
    amplitudes : np.ndarray
        Complex amplitudes.
    conjugate_pairs : list of (int, int)
        Conjugate pair indices.
    mode_indices : list of int, optional
        If given, only pairs whose *both* members appear in this set
        are included.

    Returns
    -------
    np.ndarray
        Real-valued, shape ``(n_times, n_coords)``.
    """
    n_coords = modes.shape[0]
    result = np.zeros((len(times), n_coords))
    idx_set = set(mode_indices) if mode_indices is not None else None

    for i, j in conjugate_pairs:
        # Skip pairs not in the requested subset
        if idx_set is not None and (
            (i == j and i not in idx_set)
            or (i != j and (i not in idx_set or j not in idx_set))
        ):
            continue

        omega = np.imag(eigenvalues[i])
        sign_omega = np.sign(omega) if abs(omega) > ZERO_FREQUENCY_THRESHOLD else 1.0
        magnitude = np.abs(modes[:, i])
        phase = np.arctan2(-sign_omega * np.imag(modes[:, i]), np.real(modes[:, i]))
        beta = np.abs(amplitudes[i])
        freq = abs(omega)
        factor = 1.0 if i == j else 2.0

        # Vectorised over time
        cos_term = np.cos(np.outer(times, np.array([freq])) - phase[np.newaxis, :])
        result += factor * beta * magnitude[np.newaxis, :] * cos_term

    return result


def _apply_eigenvalue_modifications(
    eigenvalues: np.ndarray,
    modify_eigenvalues: dict,
    conjugate_pairs: list,
    stable: bool,
) -> set:
    """Modify eigenvalues in place and return the set of processed indices."""
    zero_modes = modify_eigenvalues.get("zero_modes", [])
    changed_freq = modify_eigenvalues.get("changed_freq")

    conj_map = {}
    for i, j in conjugate_pairs:
        conj_map[i] = j
        conj_map[j] = i

    processed: set = set()
    for idx in zero_modes:
        if idx in processed:
            continue
        conj_idx = conj_map.get(idx, idx)

        if changed_freq is None:
            eigenvalues[idx] = 0.0 + 0.0j
            eigenvalues[conj_idx] = 0.0 + 0.0j
        elif stable:
            eigenvalues[idx] = changed_freq * 1j
            eigenvalues[conj_idx] = -changed_freq * 1j
        else:
            eigenvalues[idx] = np.real(eigenvalues[idx]) + changed_freq * 1j
            eigenvalues[conj_idx] = np.real(eigenvalues[conj_idx]) - changed_freq * 1j

        processed.update((idx, conj_idx))
    return processed


# ── Public API ──────────────────────────────────────────────────────


def reconstruct(
    result: DMDResult,
    times: np.ndarray | None = None,
    pairs: list[int] | None = None,
    mode_indices: list[int] | None = None,
    stable: bool = False,
) -> np.ndarray:
    """Rebuild the signal from a DMD decomposition.

    This single function replaces ``reconstruct_dmd``,
    ``reconstruct_dmd_complex``, ``reconstruct_specific_modes``, and
    ``reconstruct_conjugate_pairs`` from the old API.

    Parameters
    ----------
    result : DMDResult
        Output of :func:`run_dmd`.
    times : np.ndarray, optional
        Time vector.  Defaults to ``result.times[1:]`` (the training
        window minus the first frame, matching BOPDMD convention).
    pairs : list of int, optional
        Conjugate-pair numbers (0-indexed) to include.  Converted to
        mode indices internally.  Mutually exclusive with
        *mode_indices*.
    mode_indices : list of int, optional
        Raw mode indices to include.  Mutually exclusive with *pairs*.
    stable : bool
        If ``True``, use phasor reconstruction (frequencies only,
        no growth/decay).  If ``False`` (default), use full complex
        eigenvalues.

    Returns
    -------
    np.ndarray
        Reconstructed keypoints, shape ``(n_times, n_markers, 3)``,
        with the mean shape added back.

    Notes
    -----
    With ``stable=False``, uses the full eigenvalue reconstruction:

        x̃(t) = Re[ Σⱼ bⱼ φⱼ exp(λⱼ t) ]

    With ``stable=True``, strips growth/decay for bounded output:

        x̃(t) = Σ_{pairs} 2|b| |φ| cos(|ω|t - θ)

    Raises
    ------
    ValueError
        If both *pairs* and *mode_indices* are supplied.
    """
    if pairs is not None and mode_indices is not None:
        msg = "Specify *pairs* or *mode_indices*, not both"
        raise ValueError(msg)

    if times is None:
        times = result.times[1:]

    dmd = result._dmd_object
    # Ensure we only use the coordinate rows, not extra Hankel rows
    n_markers = result.mode_magnitudes.shape[0]
    n_actual_coords = n_markers * 3
    modes_trimmed = dmd.modes[:n_actual_coords, :]

    # Resolve pairs → mode_indices
    if pairs is not None:
        mode_indices = result.pair_indices(pairs)

    if stable:
        recon = _reconstruct_phasor(
            times,
            dmd.eigs,
            modes_trimmed,
            dmd.amplitudes,
            result.conjugate_pairs,
            mode_indices,
        )
    else:
        recon = _reconstruct_complex(
            times,
            dmd.eigs,
            modes_trimmed,
            dmd.amplitudes,
            mode_indices,
        )

    # Reshape and add mean shape
    avg = result.average_shape.reshape(-1, n_markers, 3)
    return recon.reshape(-1, n_markers, 3) + avg


def forecast(
    result: DMDResult,
    times: np.ndarray,
    pairs: list[int] | None = None,
    mode_indices: list[int] | None = None,
    stable: bool = True,
    modify_eigenvalues: dict | None = None,
) -> np.ndarray:
    """Extrapolate beyond the training window.

    Like :func:`reconstruct` but with additional controls for
    eigenvalue modification (e.g. changing frequencies, zeroing
    specific modes).

    Parameters
    ----------
    result : DMDResult
        Output of :func:`run_dmd`.
    times : np.ndarray
        Forecast time vector (may extend beyond training window).
    pairs : list of int, optional
        Conjugate-pair numbers to include.
    mode_indices : list of int, optional
        Raw mode indices to include.
    stable : bool
        If ``True`` (default), strip growth/decay.
    modify_eigenvalues : dict, optional
        Eigenvalue modifications.  Keys:

        * ``"zero_modes"`` — list of mode indices whose eigenvalues
          are set to zero (and their conjugates).
        * ``"changed_freq"`` — new frequency (rad/s) to assign to
          the modes in ``"zero_modes"`` instead of zero.

    Returns
    -------
    np.ndarray
        Forecast keypoints, shape ``(n_times, n_markers, 3)``.

    Notes
    -----
    This replaces ``run_forecast``, ``run_forecast_with_modified_modes``,
    and ``modify_mode_frequencies`` from the old API.
    """
    if pairs is not None and mode_indices is not None:
        msg = "Specify *pairs* or *mode_indices*, not both"
        raise ValueError(msg)

    dmd = result._dmd_object
    n_markers = result.mode_magnitudes.shape[0]
    n_coords = n_markers * 3
    modes = dmd.modes[:n_coords, :]
    amplitudes = dmd.amplitudes.copy()
    eigenvalues = dmd.eigs.copy()

    # Apply eigenvalue modifications
    modified_indices: set = set()
    if modify_eigenvalues is not None:
        modified_indices = _apply_eigenvalue_modifications(
            eigenvalues,
            modify_eigenvalues,
            result.conjugate_pairs,
            stable,
        )

    # Resolve pairs → mode_indices
    if pairs is not None:
        mode_indices = result.pair_indices(pairs)

    # Strip growth/decay if stable
    if stable and not modified_indices:
        eigenvalues = 1j * np.imag(eigenvalues)
    elif stable and modified_indices:
        # Strip real parts only from unmodified eigenvalues
        for k in range(len(eigenvalues)):
            if k not in modified_indices:
                eigenvalues[k] = 1j * np.imag(eigenvalues[k])

    # Subset modes if requested
    if mode_indices is not None:
        eigenvalues = eigenvalues[mode_indices]
        modes = modes[:, mode_indices]
        amplitudes = amplitudes[mode_indices]

    # Reconstruct
    dynamics = np.exp(np.outer(eigenvalues, times))
    weighted = modes * amplitudes[np.newaxis, :]
    recon = np.real((weighted @ dynamics).T)

    avg = result.average_shape.reshape(-1, n_markers, 3)
    return recon.reshape(-1, n_markers, 3) + avg
