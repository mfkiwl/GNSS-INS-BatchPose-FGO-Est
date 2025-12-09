import numpy as np
from typing import Dict, Optional, Tuple

from gnss_utils.gnss_data_utils import GnssMeasurementChannel, SignalChannelId
from constants.parameters import (
    BASE_POS_ECEF,
    BASE_ECEF_TO_ENU_ROT_MAT,
    GNSS_ELEV_MODEL_PARAMS,
    GnssParameters,
    phase_sigma_a,
    phase_sigma_b,
)
from gnss_utils import satellite_utils
from constants.common_utils import skew_symmetric


# Import the function from constants to avoid circular import
def compute_ecef_ned_rot_mat(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Import the function from common_utils to avoid circular import."""
    from constants.common_utils import (
        compute_ecef_ned_rot_mat as _compute_ecef_ned_rot_mat,
    )

    return _compute_ecef_ned_rot_mat(lat_rad, lon_rad)


def compute_ecef_enu_rot_mat(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Import the function from common_utils to avoid circular import."""
    from constants.common_utils import (
        compute_ecef_enu_rot_mat as _compute_ecef_enu_rot_mat,
    )

    return _compute_ecef_enu_rot_mat(lat_rad, lon_rad)


def compute_world_frame_coord_from_ecef(ecef_pos_m: np.ndarray) -> np.ndarray:
    """Convert ECEF coordinates to local ENU frame coordinates.
    World frame is defined as ENU frame centered at the base station.
    """

    # Apply the rotation to the ECEF position
    enu_pos_m = BASE_ECEF_TO_ENU_ROT_MAT() @ (ecef_pos_m - BASE_POS_ECEF)

    return enu_pos_m


def cn0_based_noise_std(cn0: float, a: float, b: float) -> float:
    """Compute noise standard deviation based on C/N0 value.

    Args:
        cn0: C/N0 value in dB-Hz.
        a: Parameter a in the noise model.
        b: Parameter b in the noise model.

    Returns:
        Noise standard deviation.
    """

    # a + b * 10^(-cn0/10)
    return a + b * 10 ** (-cn0 / 10)


def elevation_based_noise_var(elev_deg: float, sigma_a: float, sigma_b: float) -> float:
    """Compute noise variance based on elevation angle.
    Args:
        elev_deg: Elevation angle in degrees.
        a: Parameter a in the noise model.
        b: Parameter b in the noise model.
    Returns:
        Noise standard deviation.
    """

    # a + b / sin(elev)
    elev_rad = np.radians(elev_deg)
    sin_elev = np.sin(elev_rad)
    if sin_elev < 1e-6:
        sin_elev = 1e-6  # Prevent division by zero or very small values

    return 2 * sigma_a**2 + (sigma_b / sin_elev) ** 2


def elevation_based_noise_var(elev_deg: float, a: float, b: float) -> float:
    """Compute noise variance based on elevation angle.

    Args:
        elev_deg: Elevation angle in degrees.
        a: Parameter a in the noise model.
        b: Parameter b in the noise model.

    Returns:
        Noise standard deviation.
    """

    # a + b / sin(elev)
    elev_rad = np.radians(elev_deg)
    sin_elev = np.sin(elev_rad)
    if sin_elev < 1e-6:
        sin_elev = 1e-6  # Prevent division by zero or very small values

    return 2 * a**2 + (b / sin_elev) ** 2


def select_pivot_satellite(
    channels: Dict[SignalChannelId, GnssMeasurementChannel],
    recv_pos_ecef: np.ndarray,
    ecef_to_enu_rot: np.ndarray,
    *,
    min_elev_deg: Optional[float] = None,
    min_cn0_dbhz: Optional[float] = None,
    verbose_prefix: str = "",
) -> Optional[Tuple[SignalChannelId, GnssMeasurementChannel, float]]:
    """
    Select a pivot satellite among the provided channels using CN0 priority,
    then elevation as tiebreaker.

    A valid pivot must satisfy elevation >= ``min_elev_deg`` and
    CN0 >= ``min_cn0_dbhz``. Among valid candidates, choose the highest CN0;
    if multiple share the same CN0, choose the one with the higher elevation.

    If fewer than 3 channels are available or no candidate passes the
    thresholds, return None and optionally print a short message.

    Returns a tuple of (SignalChannelId, channel, elevation_deg) if selected.
    """

    if min_elev_deg is None or min_cn0_dbhz is None:
        # Lazy import to avoid circular dependency at module import time
        from constants.parameters import GnssParameters as _GP

        if min_elev_deg is None:
            min_elev_deg = _GP.PIVOT_SAT_ELEVATION_MASK_DEG
        if min_cn0_dbhz is None:
            min_cn0_dbhz = _GP.PIVOT_SAT_CNO_THRESHOLD

    if channels is None or len(channels) < 3:
        if verbose_prefix:
            print(f"{verbose_prefix}: less than 3 satellites, skip epoch")
        return None

    candidates: list[Tuple[SignalChannelId, GnssMeasurementChannel, float]] = []
    for scid, ch in channels.items():
        # Require CN0 for prioritization; skip if missing
        if ch.cn0_dbhz is None:
            continue

        # Reuse pre-computed elevation/azimuth if available; otherwise compute and cache
        elev_deg = ch.elevation_deg
        if elev_deg is None:
            if ch.sat_pos_ecef_m is None:
                continue
            elev_deg, az_deg = satellite_utils.compute_sat_elev_az(
                ecef_to_enu_rot, recv_pos_ecef, ch.sat_pos_ecef_m
            )
            ch.elevation_deg = elev_deg
            ch.azimuth_deg = az_deg

        if elev_deg >= min_elev_deg and ch.cn0_dbhz >= min_cn0_dbhz:
            candidates.append((scid, ch, elev_deg))

    if not candidates:
        if verbose_prefix:
            print(
                f"{verbose_prefix}: no pivot candidate above thresholds (elev>={min_elev_deg} deg, CN0>={min_cn0_dbhz} dB-Hz)"
            )
        return None

    # Select by CN0 (desc), then elevation (desc)
    candidates.sort(key=lambda t: (t[1].cn0_dbhz, t[2]), reverse=True)
    return candidates[0]


def measurement_std(signal_type, elev_deg: float) -> tuple[float, float]:
    params = GNSS_ELEV_MODEL_PARAMS.get(signal_type)
    if params is None:
        raise ValueError(f"No elevation model sigma parameters for {signal_type}")
    code_var = elevation_based_noise_var(elev_deg, params[0], params[1])
    phase_var = elevation_based_noise_var(elev_deg, phase_sigma_a, 2.0 * phase_sigma_b)
    return float(np.sqrt(code_var)), float(np.sqrt(phase_var))


def doppler_meas_variance(
    los_ecef: np.ndarray,
    rot_enu_from_body: np.ndarray,
    rot_ecef_from_enu: np.ndarray,
    lever_arm_b: np.ndarray,
    gyro_noise_std: np.ndarray,
) -> float:
    chain = (
        los_ecef @ rot_ecef_from_enu @ rot_enu_from_body @ skew_symmetric(lever_arm_b)
    )
    sigma_gyro = float(chain @ gyro_noise_std)
    return sigma_gyro**2 + GnssParameters.DOPPLER_SIGMA_MPS**2
