from dataclasses import dataclass, field
import numpy as np
import pymap3d as pm


def computeGravityConst(lat: float) -> float:
    g_h = (
        9.7803267715
        * (1 + 0.001931851353 * (np.sin(lat)) ** 2)
        / np.sqrt(1 - 0.0066943800229 * (np.sin(lat)) ** 2)
    )

    return g_h


@dataclass
class GnssParameters:
    ELEVATION_MASK: float = 15.0  # Minimum elevation angle in degrees
    CNO_THRESHOLD: float = (
        20.0  # Minimum C/N0 in dB-Hz for a satellite to be considered valid
    )

    enable_gps: bool = True
    enable_galileo: bool = True
    enable_glonass: bool = True
    enable_beidou: bool = True


RINEX_OBS_CHANNEL_TO_USE: dict[str, set[str]] = {
    "G": {"1C", "2L"},
    "R": {"1C", "2C"},
    "E": {"1C", "7Q"},
    "C": {"2I"},
}

BASE_POS_ECEF = [
    -742080.4125,
    -5462031.7412,
    3198339.6909,
]  # Base station position in ECEF coordinates (meters)

BASE_POS_LLA = pm.ecef2geodetic(
    BASE_POS_ECEF[0], BASE_POS_ECEF[1], BASE_POS_ECEF[2]
)  # Base station position in LLA (lat, lon, alt)


INCH_TO_METER = 0.0254  # 1 inch = 0.0254 meters


@dataclass
class TexCupBoschImuParams:
    # GNSS antenna to IMU translation (meters) in IMU body frame.
    t_ant_to_imu_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [16.1811 * INCH_TO_METER, -6.2992 * INCH_TO_METER, 4.8425 * INCH_TO_METER]
        )
    )

    gravity: float = computeGravityConst(np.deg2rad(BASE_POS_LLA[0]))
    z_up: bool = True  # IMU z-axis points upward (GTSAM assumes Z axis pointing up)

    acc_noise_std = np.full(
        3, np.sqrt(0.1) * gravity / 1e3
    )  # 0.1 mg^2/Hz > m/s^2/sqrt(Hz)
    acc_bias_std = np.full(3, 10 * gravity / 1e3)  # 10 milli-g > m/s^2
    gyro_noise_std = np.full(
        3,
        np.sqrt(0.02) * np.pi / 180,
    )  # 0.02 (deg/s)^2/Hz -> rad/s/sqrt(Hz)
    gyro_bias_std = np.full(
        3,
        200 * np.pi / 180 / 3600,
    )  # 200 deg/h -> rad/s


@dataclass
class TexCupLordImuParams:
    # GNSS antenna to IMU translation (meters) in IMU body frame.
    t_ant_to_imu_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.461 * INCH_TO_METER, -0.125 * INCH_TO_METER, -0.119 * INCH_TO_METER]
        )
    )

    gravity: float = computeGravityConst(np.deg2rad(BASE_POS_LLA[0]))
    z_up: bool = False

    acc_noise_std = np.full(3, 0.1 * gravity / 1e3)  # 0.1 mg/sqrt(Hz) > m/s^2/sqrt(Hz)
    acc_bias_std = np.full(3, 0.5 * gravity / 1e3)  # 0.5 milli-g > m/s^2
    gyro_noise_std = np.full(
        3,
        0.01 * np.pi / 180,
    )  # 0.01 deg/s/sqrt(Hz) -> rad/s/sqrt(Hz)
    gyro_bias_std = np.full(
        3,
        8 * np.pi / 180 / 3600,
    )  # 8 deg/h -> rad/s
