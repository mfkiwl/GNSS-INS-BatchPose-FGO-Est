"""Analytic GNSS measurement error models used by the FGO solver."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import gtsam
import numpy as np

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
    J_theta_map = rot_enu_from_body @ utils.skew_symmetric(lever_arm_b)

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
