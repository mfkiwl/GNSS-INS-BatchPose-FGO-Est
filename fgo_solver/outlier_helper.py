"""Helpers for detecting GNSS measurement outliers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def neyman_pearson_residual_test(
    threshold: float,
    cov_prior: Optional[np.ndarray],
    residual: np.ndarray,
    H: Optional[np.ndarray],
    noise_variances: Optional[np.ndarray],
) -> np.ndarray:
    """Return a boolean mask indicating which residual components pass the test.

    Args:
        threshold: Scalar multiplier applied to the predicted residual sigma.
        cov_prior: Prior covariance matrix for the linearised state (NxN).
        residual: Residual vector ``z - h(x)`` (length M).
        H: Linearised measurement Jacobian wrt the state (MxN).
        noise_variances: Diagonal of the measurement noise covariance (length M).

    Returns:
        Boolean array of length ``len(residual)`` where ``True`` indicates the
        residual component is accepted (kept) and ``False`` indicates rejection.
        If ``cov_prior``, ``H`` or ``noise_variances`` are missing, all residuals
        are accepted.
    """

    res = np.atleast_1d(np.asarray(residual, dtype=float))
    length = res.shape[0]

    if (
        cov_prior is None
        or H is None
        or noise_variances is None
        or length == 0
    ):
        return np.ones(length, dtype=bool)

    cov = np.asarray(cov_prior, dtype=float)
    jac = np.asarray(H, dtype=float)
    noise = np.atleast_1d(np.asarray(noise_variances, dtype=float))

    if jac.shape[0] != length:
        raise ValueError("Jacobian row count must match residual length.")
    if jac.shape[1] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        raise ValueError("Prior covariance must be square and align with H.")
    if noise.ndim != 1 or noise.shape[0] != length:
        raise ValueError("Noise variances must be a 1-D array matching residuals.")

    projected = jac @ cov
    pred_var = np.einsum("ij,ij->i", projected, jac) + noise
    pred_var = np.maximum(pred_var, 0.0)

    sigma = threshold * np.sqrt(pred_var)
    return np.abs(res) <= sigma
