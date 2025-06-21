from dataclasses import dataclass


@dataclass
class GnssParameters:
    elevation_mask: float = 15.0  # Minimum elevation angle in degrees
    cn0_threshold: float = (
        20.0  # Minimum C/N0 in dB-Hz for a satellite to be considered valid
    )
