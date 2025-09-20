import numpy as np


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


def compute_ecef_ned_rot_mat(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    Compute rotation matrix from ECEF to NED frame at given latitude and longitude.

    Args:
        lat_rad: Latitude in radians.
        lon_rad: Longitude in radians.

    Returns:
        Rotation matrix (3x3 numpy array) from ECEF to NED.
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Rotation from ECEF to NED
    C_e_n = np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ]
    )
    return C_e_n
