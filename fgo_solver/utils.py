from typing import Iterable, Set, Tuple

import numpy as np

from constants.gnss_constants import Constellation
from gnss_utils.gnss_data_utils import GnssMeasurementChannel


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Convert 3D vector to skew-symmetric matrix."""
    if v.shape != (3,):
        raise ValueError("Input vector must be 3-dimensional.")
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def extract_satellite_id(
    channel: GnssMeasurementChannel,
) -> Tuple[Constellation, int]:
    """Return (constellation, PRN) tuple for a measurement channel."""
    sat_id = channel.signal_id.satellite_id
    return sat_id.constellation, sat_id.prn


def unique_satellite_ids(
    channels: Iterable[GnssMeasurementChannel],
) -> Set[Tuple[Constellation, int]]:
    """Collect unique satellites represented in the provided channels."""
    unique: Set[Tuple[Constellation, int]] = set()
    for channel in channels:
        if channel.signal_id is None or channel.signal_id.satellite_id is None:
            continue
        unique.add(extract_satellite_id(channel))
    return unique
