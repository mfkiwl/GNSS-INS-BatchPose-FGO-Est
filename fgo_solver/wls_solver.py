"""Robust double-differenced GNSS WLS position solver."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from constants.parameters import (
    BASE_POS_ECEF,
    BASE_ECEF_TO_ENU_ROT_MAT,
    GNSS_ELEV_MODEL_PARAMS,
    GnssParameters,
    phase_sigma_a,
    phase_sigma_b,
)
from gnss_utils import satellite_utils
from gnss_utils.gnss_data_utils import (
    GnssMeasurementChannel,
    SignalChannelId,
)
from gnss_utils.gnss_dataclass import SatelliteId, SignalType
from gnss_utils.model_utils import elevation_based_noise_var, select_pivot_satellite
from gnss_utils.satellite_utils import OMEGA_E_DIV_C, sagnac_correction

_DEFAULT_MAX_ITERS = 10
_POS_TOL_M = 1e-3
_DEFAULT_HUBER_K = GnssParameters().HUBER_K
_DEFAULT_CODE_SIGMA_M = 1.0
_DEFAULT_PHASE_SIGMA_M = 0.02


def solve_wls_position(
    channels: Dict[SignalChannelId, GnssMeasurementChannel],
    *,
    max_iters: int = _DEFAULT_MAX_ITERS,
    huber_k: float = _DEFAULT_HUBER_K,
    position_convergence_tol_m: float = _POS_TOL_M,
) -> Tuple[np.ndarray, float]:
    """Estimate receiver ECEF position and double-differenced ambiguities.

    Carrier-phase ambiguities are explicitly modelled while clock bias is
    eliminated through double differencing. Both code and carrier-phase
    measurements contribute to the solution, and a Huber M-estimator provides
    robustness against outliers.
    """
    est = np.asarray(BASE_POS_ECEF, dtype=float)

    grouped = _group_channels_by_signal(channels)
    if not grouped:
        raise ValueError("No valid GNSS channels available for WLS initialization")

    ecef_to_enu_rot = BASE_ECEF_TO_ENU_ROT_MAT()

    pivot_data = {}
    meas_entries: List[dict] = []
    ambiguity_init: List[float] = []
    amb_index_map: dict[Tuple[SignalType, SatelliteId], int] = {}

    for signal_type, group_channels in grouped.items():
        pivot_info = select_pivot_satellite(group_channels, est, ecef_to_enu_rot)
        if pivot_info is None:
            continue

        pivot_id, pivot_ch, _ = pivot_info
        if (
            pivot_ch.code_m is None
            or pivot_ch.phase_m is None
            or pivot_ch.sat_pos_ecef_m is None
        ):
            continue

        entries = []
        for scid, ch in group_channels.items():
            if scid == pivot_id:
                continue
            if (
                ch.code_m is None
                or ch.phase_m is None
                or ch.sat_pos_ecef_m is None
                or pivot_ch.sat_pos_ecef_m is None
                or ch.wavelength_m is None
                or ch.wavelength_m == 0.0
            ):
                continue

            key = (signal_type, scid.satellite_id)
            amb_idx = amb_index_map.get(key)
            if amb_idx is None:
                amb_idx = len(ambiguity_init)
                amb_index_map[key] = amb_idx
                ambiguity_init.append(
                    (ch.phase_m - pivot_ch.phase_m - (ch.code_m - pivot_ch.code_m))
                    / ch.wavelength_m
                )

            entries.append(
                {
                    "signal_type": signal_type,
                    "channel": ch,
                    "pivot": pivot_ch,
                    "amb_idx": amb_idx,
                    "code_diff": ch.code_m - pivot_ch.code_m,
                    "phase_diff": ch.phase_m - pivot_ch.phase_m,
                }
            )

        if entries:
            pivot_data[signal_type] = pivot_ch
            meas_entries.extend(entries)

    if len(ambiguity_init) < 3:
        raise ValueError(
            "Insufficient satellites for double-difference WLS initialization"
        )

    num_state = 3 + len(ambiguity_init)
    amb_state = np.asarray(ambiguity_init, dtype=float)

    for _ in range(max_iters):
        pivot_geom = {}
        for signal_type, pivot_ch in pivot_data.items():
            if pivot_ch.sat_pos_ecef_m is None:
                continue
            vec_pivot = est - pivot_ch.sat_pos_ecef_m
            pivot_range = np.linalg.norm(vec_pivot)
            if pivot_range < 1.0:
                raise ValueError("Invalid geometry for pivot satellite")
            pivot_unit = vec_pivot / pivot_range
            pivot_range += sagnac_correction(est, pivot_ch.sat_pos_ecef_m)
            pivot_geom[signal_type] = (pivot_range, pivot_unit)

        design_rows: List[np.ndarray] = []
        residuals: List[float] = []
        residual_types: List[str] = []

        for entry in meas_entries:
            signal_type = entry["signal_type"]
            pivot_info = pivot_geom.get(signal_type)
            if pivot_info is None:
                continue
            pivot_range, unit_p = pivot_info

            channel = entry["channel"]
            pivot_ch = entry["pivot"]
            if channel.sat_pos_ecef_m is None or pivot_ch.sat_pos_ecef_m is None:
                continue

            vec_sat = est - channel.sat_pos_ecef_m
            sat_range = np.linalg.norm(vec_sat)
            if sat_range < 1.0:
                continue

            unit_s = vec_sat / sat_range
            sat_range += sagnac_correction(est, channel.sat_pos_ecef_m)

            grad_sagnac_sat = OMEGA_E_DIV_C * np.array(
                [-channel.sat_pos_ecef_m[1], channel.sat_pos_ecef_m[0], 0.0]
            )
            grad_sagnac_pivot = OMEGA_E_DIV_C * np.array(
                [-pivot_ch.sat_pos_ecef_m[1], pivot_ch.sat_pos_ecef_m[0], 0.0]
            )

            geom = unit_s - unit_p + grad_sagnac_sat - grad_sagnac_pivot
            pred_dd = sat_range - pivot_range
            amb_idx = entry["amb_idx"]

            code_sigma, phase_sigma = _compute_dd_sigmas(
                signal_type, channel, pivot_ch, est, ecef_to_enu_rot
            )

            if code_sigma <= 0.0:
                raise ValueError("Invalid code sigma")

            code_row = np.zeros(num_state, dtype=float)
            code_row[:3] = geom / code_sigma
            code_res = (entry["code_diff"] - pred_dd) / code_sigma
            design_rows.append(code_row)
            residuals.append(code_res)
            residual_types.append("code")

            if phase_sigma <= 0.0:
                raise ValueError("Invalid phase sigma")

            phase_row = np.zeros(num_state, dtype=float)
            phase_row[:3] = geom / phase_sigma
            phase_row[3 + amb_idx] = channel.wavelength_m / phase_sigma
            phase_res = (
                entry["phase_diff"]
                - pred_dd
                - channel.wavelength_m * amb_state[amb_idx]
            ) / phase_sigma
            design_rows.append(phase_row)
            residuals.append(phase_res)
            residual_types.append("phase")

        if len(design_rows) < num_state:
            raise ValueError("Insufficient measurement geometry for WLS convergence")

        design_matrix = np.vstack(design_rows)
        residual_vec = np.asarray(residuals)

        weights = np.ones_like(residual_vec)
        residual_types_arr = np.asarray(residual_types)
        code_mask = residual_types_arr == "code"
        if np.any(code_mask):
            code_residuals = residual_vec[code_mask]
            weights[code_mask] = _compute_huber_weights(code_residuals, huber_k)
        phase_mask = residual_types_arr == "phase"
        if np.any(phase_mask):
            phase_residuals = residual_vec[phase_mask]
            weights[phase_mask] = _compute_huber_weights(phase_residuals, huber_k)

        sqrt_weights = np.sqrt(weights)
        weighted_design = design_matrix * sqrt_weights[:, None]
        weighted_residual = residual_vec * sqrt_weights

        dx, *_ = np.linalg.lstsq(weighted_design, weighted_residual, rcond=None)
        est += dx[:3]
        if num_state > 3:
            amb_state += dx[3:]

        if np.linalg.norm(dx[:3]) < position_convergence_tol_m:
            break

    return est, 0.0


def _compute_huber_weights(residuals: np.ndarray, huber_k: float) -> np.ndarray:
    """Return per-measurement weights using a Huber M-estimator."""
    if residuals.size == 0:
        return np.array([], dtype=float)

    scale = _robust_scale(residuals)
    if scale <= 0.0 or huber_k <= 0.0:
        return np.ones_like(residuals)

    threshold = huber_k * scale
    abs_residuals = np.abs(residuals)
    weights = np.ones_like(residuals)
    mask = abs_residuals > threshold
    weights[mask] = threshold / abs_residuals[mask]
    return weights


def _robust_scale(residuals: np.ndarray) -> float:
    """Estimate residual scale using MAD for numerical robustness."""
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    if mad > 1e-9:
        return 1.4826 * mad

    std = np.std(residuals)
    if std > 0.0:
        return std

    return 1.0


def _group_channels_by_signal(
    channels: Dict[SignalChannelId, GnssMeasurementChannel],
) -> Dict:
    grouped: Dict = {}
    for scid, ch in channels.items():
        signal_type = scid.signal_type
        if signal_type not in grouped:
            grouped[signal_type] = {}
        grouped[signal_type][scid] = ch
    return grouped


def _compute_dd_sigmas(
    signal_type: SignalType,
    channel: GnssMeasurementChannel,
    pivot_ch: GnssMeasurementChannel,
    recv_pos_ecef: np.ndarray,
    ecef_to_enu_rot: np.ndarray,
) -> Tuple[float, float]:
    code_std_sat, phase_std_sat = _single_measurement_std(
        signal_type, channel, recv_pos_ecef, ecef_to_enu_rot
    )
    code_std_pivot, phase_std_pivot = _single_measurement_std(
        signal_type, pivot_ch, recv_pos_ecef, ecef_to_enu_rot
    )
    code_sigma = float(np.hypot(code_std_sat, code_std_pivot))
    phase_sigma = float(np.hypot(phase_std_sat, phase_std_pivot))
    return code_sigma, phase_sigma


def _single_measurement_std(
    signal_type: SignalType,
    channel: GnssMeasurementChannel,
    recv_pos_ecef: np.ndarray,
    ecef_to_enu_rot: np.ndarray,
) -> Tuple[float, float]:
    elev_deg = _elevation_deg(channel, recv_pos_ecef, ecef_to_enu_rot)

    params = GNSS_ELEV_MODEL_PARAMS.get(signal_type)
    if params is None:
        raise ValueError(f"No elevation noise parameters for signal {signal_type}")

    code_var = elevation_based_noise_var(elev_deg, params[0], params[1])
    phase_var = elevation_based_noise_var(elev_deg, phase_sigma_a, 2.0 * phase_sigma_b)
    code_std = float(np.sqrt(max(code_var, 0.0)))
    phase_std = float(np.sqrt(max(phase_var, 0.0)))
    return code_std, phase_std


def _elevation_deg(
    channel: GnssMeasurementChannel,
    recv_pos_ecef: np.ndarray,
    ecef_to_enu_rot: np.ndarray,
) -> float:
    if channel.sat_pos_ecef_m is None:
        raise ValueError("Satellite position is required to compute elevation")
    elev_deg, az_deg = satellite_utils.compute_sat_elev_az(
        ecef_to_enu_rot, recv_pos_ecef, channel.sat_pos_ecef_m
    )
    channel.elevation_deg = elev_deg
    channel.azimuth_deg = az_deg
    return elev_deg
