import numpy as np


def compute_ecef_ned_rot_mat(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    Compute the rotation matrix from ECEF to NED frame.

    Args:
        lat_rad: Latitude in radians
        lon_rad: Longitude in radians

    Returns:
        3x3 rotation matrix from ECEF to NED
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    return np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ]
    )


def compute_ecef_enu_rot_mat(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    Compute the rotation matrix from ECEF to ENU frame.

    Args:
        lat_rad: Latitude in radians
        lon_rad: Longitude in radians

    Returns:
        3x3 rotation matrix from ECEF to ENU
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    return np.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def computeGravityConst(lat_rad: float) -> float:
    g_h = (
        9.7803267715
        * (1 + 0.001931851353 * (np.sin(lat_rad)) ** 2)
        / np.sqrt(1 - 0.0066943800229 * (np.sin(lat_rad)) ** 2)
    )

    return g_h
