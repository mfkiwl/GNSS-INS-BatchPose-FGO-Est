"""Analytic GNSS measurement error models used by the FGO solver."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import gtsam
import numpy as np

from constants.gnss_constants import EarthConstants, SPEED_OF_LIGHT_MS
from constants.common_utils import rotation_z, skew_symmetric
from constants.parameters import BASE_ECEF_TO_ENU_ROT_MAT, BASE_POS_ECEF
from fgo_solver import utils
from gnss_utils.gnss_data_utils import GnssMeasurementChannel
from gnss_utils.satellite_utils import OMEGA_E_DIV_C, sagnac_correction


def _antenna_ecef_from_pose(
    pose: gtsam.Pose3, lever_arm_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return antenna ECEF position and helper transforms.

    Args:
        pose: Current IMU pose (body in ENU frame).
        lever_arm_b: Lever arm vector in IMU body coordinates (meters).

    Returns:
        Tuple containing antenna ECEF coordinates, orientation (body->ENU), and
        ENU->ECEF rotation.
    """

    rot_ecef_from_enu = BASE_ECEF_TO_ENU_ROT_MAT().T
    rot_enu_from_body = pose.rotation().matrix()
    lever_enu = rot_enu_from_body @ lever_arm_b
    antenna_ecef = rot_ecef_from_enu @ (pose.translation() + lever_enu) + np.asarray(
        BASE_POS_ECEF
    )
    return antenna_ecef, rot_enu_from_body, rot_ecef_from_enu


def double_difference_batch_residual_and_jacobian(
    pivot: GnssMeasurementChannel,
    measurements: Sequence[GnssMeasurementChannel],
    pose: gtsam.Pose3,
    lever_arm_b: np.ndarray,
    amb_vals: Optional[Iterable[float]] = None,
    compute_jacobian: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """Compute double-difference residuals (and optional Jacobians).

    The residual is ``measurement - prediction`` for code/phase double
    differences, linearised around the provided pose/ambiguity state.

    Parameters
    ----------
    pivot : GnssMeasurementChannel
        The pivot satellite channel providing the shared reference.
    measurements : Sequence[GnssMeasurementChannel]
        Other channels within the same signal type; one ambiguity per entry.
    pose : gtsam.Pose3
        IMU pose expressed in ENU (world<-body). Perturbations are in the
        body frame, matching GTSAM's left-multiplicative convention.
    lever_arm_b : np.ndarray
        Lever arm from IMU to antenna in body coordinates (meters).
    amb_vals : Iterable[float]
        Double-differenced ambiguities (cycles) aligned with ``measurements``.
    compute_jacobian : bool
        When True, return analytic Jacobians wrt pose and ambiguities.

    Returns
    -------
    residual : np.ndarray
        Stacked vector ``[code_dd; phase_dd]`` of length ``2N``.
    J_pose : np.ndarray or None
        ``(2N x 6)`` Jacobian wrt pose (rotation then translation). ``None``
        if ``compute_jacobian`` is False.
    J_amb_list : list[np.ndarray] or None
        List of ``(2N x 1)`` columns for each ambiguity. ``None`` otherwise.
    """

    n = len(measurements)
    residual = np.zeros(2 * n, dtype=float)
    if n == 0:
        return residual, None, None

    if amb_vals is None or len(amb_vals) != n:
        raise ValueError(
            "Ambiguity values must be provided for every measurement in the batch"
        )
    amb_list = list(amb_vals)

    ant_ecef, rot_enu_from_body, rot_ecef_from_enu = _antenna_ecef_from_pose(
        pose, lever_arm_b
    )

    # Pivot geometry.
    vec_pivot = ant_ecef - pivot.sat_pos_ecef_m
    est_range_pivot = np.linalg.norm(vec_pivot)
    if est_range_pivot < 1e-9:
        raise ValueError("Pivot satellite range is ill-conditioned")
    los_pivot = vec_pivot / est_range_pivot
    est_range_pivot += sagnac_correction(ant_ecef, pivot.sat_pos_ecef_m)

    # Lever arm contribution for the attitude block:
    #  d(R * t_lever)/d theta  |_{delta=0} = R * [t_lever]_x
    J_theta_map = rot_enu_from_body @ skew_symmetric(lever_arm_b)

    J_pose = np.zeros((2 * n, 6), dtype=float) if compute_jacobian else None
    J_amb_list: Optional[List[np.ndarray]] = [] if compute_jacobian else None

    for idx, (meas, amb) in enumerate(zip(measurements, amb_list)):
        vec_meas = ant_ecef - meas.sat_pos_ecef_m
        est_range_meas = np.linalg.norm(vec_meas)
        if est_range_meas < 1e-9:
            raise ValueError("Satellite range is ill-conditioned")
        los_meas = vec_meas / est_range_meas
        est_range_meas += sagnac_correction(ant_ecef, meas.sat_pos_ecef_m)

        # Double-differenced residuals (code / phase):
        # r = (z - h), so the prediction enters with negative sign.
        residual[idx] = meas.code_m - est_range_meas - (pivot.code_m - est_range_pivot)
        residual[n + idx] = (
            meas.phase_m
            - est_range_meas
            - (pivot.phase_m - est_range_pivot)
            - meas.wavelength_m * amb
        )

        # Residual_code = rho - rho(T_imu_in_enu)
        #          = los_ecef * rot_ecef_from_enu * [delta_T_imu_in_enu - rot_enu_frombody * [lever_arm_body]_x * delta_theta].
        # Residual_phase = Residual_code + lambda * delta_N

        if compute_jacobian:
            grad_sagnac_meas = OMEGA_E_DIV_C * np.array(
                [-meas.sat_pos_ecef_m[1], meas.sat_pos_ecef_m[0], 0.0]
            )
            grad_sagnac_pivot = OMEGA_E_DIV_C * np.array(
                [-pivot.sat_pos_ecef_m[1], pivot.sat_pos_ecef_m[0], 0.0]
            )
            # Gradient of (rho_i - rho_pivot) wrt antenna position (ECEF).
            geo_grad = (
                los_meas - los_pivot + grad_sagnac_meas - grad_sagnac_pivot
            ).reshape(1, 3)

            # Map the ECEF gradient into the pose translation coordinates.
            # Pose perturbations are in the IMU/body frame, so we rotate from
            # ECEF -> ENU -> body and keep the leading minus to account for r = z - h.
            J_pos = -(geo_grad @ rot_ecef_from_enu @ rot_enu_from_body)

            # Rotation block: lever-arm contribution.  A left-multiplicative
            # pose perturbation results in d(R*t) = R*[t]_x * delta_theta.
            J_rot = geo_grad @ rot_ecef_from_enu @ J_theta_map
            J_row = np.hstack([J_rot, J_pos]).astype(float)

            J_pose[idx, :] = J_row
            J_pose[n + idx, :] = J_row

            J_amb = np.zeros((2 * n, 1), dtype=float)
            J_amb[n + idx, 0] = -meas.wavelength_m
            J_amb_list.append(J_amb)

    return residual, J_pose, J_amb_list


def compute_los_and_theta_with_sagnac(
    ant_ecef: np.ndarray, sat_pos_ecef: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute LOS (with Sagnac in range) and Earth-rotation angle theta."""

    vec = ant_ecef - sat_pos_ecef
    rng = np.linalg.norm(vec)
    if rng < 1e-9:
        raise ValueError("Satellite range is ill-conditioned")
    rng += sagnac_correction(ant_ecef, sat_pos_ecef)
    los = vec / rng
    theta = rng / SPEED_OF_LIGHT_MS * EarthConstants.OMEGA_E_RAD_PER_SEC
    return los, theta


def doppler_batch_residual_and_jacobian(
    pivot: GnssMeasurementChannel,
    measurements: Sequence[GnssMeasurementChannel],
    pose: gtsam.Pose3,
    velocity_enu: np.ndarray,
    lever_arm_b: np.ndarray,
    bias: gtsam.imuBias.ConstantBias,
    gyro_meas: Optional[np.ndarray],
    compute_jacobian: bool,
) -> Tuple[
    np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    """Doppler double-difference residuals and Jacobians."""

    if not measurements:
        return np.zeros(0), None, None, None

    rot_enu_from_body = pose.rotation().matrix()
    rot_ecef_from_enu = BASE_ECEF_TO_ENU_ROT_MAT().T
    ant_ecef = rot_ecef_from_enu @ (
        pose.translation() + rot_enu_from_body @ lever_arm_b
    ) + np.asarray(BASE_POS_ECEF)

    pivot_los, pivot_theta = compute_los_and_theta_with_sagnac(
        ant_ecef, pivot.sat_pos_ecef_m
    )
    pivot_sat_rot = rotation_z(pivot_theta) @ pivot.sat_vel_ecef_m
    pivot_corr = pivot.doppler_mps + pivot_los @ pivot_sat_rot

    gyro_bias = np.asarray(bias.gyroscope(), dtype=float)
    gyro_vec = np.zeros(3, dtype=float) if gyro_meas is None else np.asarray(gyro_meas)
    omega_corr = (
        gyro_vec
        - gyro_bias
        - rot_enu_from_body.T @ rot_ecef_from_enu.T @ EarthConstants.OMEGA_IEE_VEC
    )
    w_cross_lever = np.cross(omega_corr, lever_arm_b)
    lever_vel_enu = rot_enu_from_body @ w_cross_lever
    vel_ant_ecef = rot_ecef_from_enu @ (velocity_enu + lever_vel_enu)

    n = len(measurements)
    residuals = np.zeros(n, dtype=float)
    J_pose = np.zeros((n, 6), dtype=float) if compute_jacobian else None
    J_vel = np.zeros((n, 3), dtype=float) if compute_jacobian else None
    J_bias = np.zeros((n, 6), dtype=float) if compute_jacobian else None

    skew_w_lever = skew_symmetric(w_cross_lever)
    skew_lever = skew_symmetric(lever_arm_b)

    for idx, ch in enumerate(measurements):
        los, theta = compute_los_and_theta_with_sagnac(ant_ecef, ch.sat_pos_ecef_m)
        sat_rot = rotation_z(theta) @ ch.sat_vel_ecef_m
        doppler_corr = ch.doppler_mps + los @ sat_rot
        los_diff = los - pivot_los
        pred = los_diff @ vel_ant_ecef
        residuals[idx] = (doppler_corr - pivot_corr) - pred

        if compute_jacobian:
            los_chain = los_diff @ rot_ecef_from_enu
            J_vel[idx, :] = -los_chain
            J_att = -los_chain @ rot_enu_from_body @ skew_w_lever
            J_pose[idx, :3] = J_att
            J_bias[idx, 3:] = -los_chain @ rot_enu_from_body @ skew_lever

    return residuals, J_pose, J_vel, J_bias
