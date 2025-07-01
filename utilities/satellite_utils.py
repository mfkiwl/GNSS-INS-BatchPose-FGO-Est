"""Utility functions for satellite position, velocity and clock computations."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

import constants.gnss_constants as gnssConst
from utilities.gnss_data_utils import (
    Constellation,
    CdmaEphemeris,
    EphemerisData,
    GpsEphemeris,
    GloEphemeris,
    GalEphemeris,
    BdsEphemeris,
    GnssMeasurementChannel,
    SignalChannelId,
)
from utilities.time_utils import GpsTime


def _kepler_anomaly(ecc: float, mean_anom: float) -> float:
    """Solve Kepler's equation for eccentric anomaly."""
    E = mean_anom
    for i in range(30):
        dE = (E - ecc * math.sin(E) - mean_anom) / (1 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < 1e-13:
            break
    else:
        raise RuntimeError("Kepler's equation did not converge within 30 iterations")
    return E


def _compute_anomaly(tk: float, eph: CdmaEphemeris, mu: float) -> float:
    """Compute eccentric anomaly from time since ephemeris reference epoch."""
    A = eph.sqrtA**2
    n0 = math.sqrt(mu / A**3)
    n = n0 + eph.delta_n
    Mk = eph.m0 + n * tk
    Ek = _kepler_anomaly(eph.ecc, Mk)
    return n, Ek


def _broadcast_orbit_pos(
    tk: float,  # Time from ephemeris reference epoch
    eph: CdmaEphemeris,
    mu: float,
    omega_dot_e: float,
    const: Constellation,
) -> Tuple[np.ndarray, float]:
    """Return satellite position and eccentric anomaly."""

    A = eph.sqrtA**2
    _, Ek = _compute_anomaly(tk, eph, mu)
    sinE = math.sin(Ek)
    cosE = math.cos(Ek)
    sqrt1_e2 = math.sqrt(1.0 - eph.ecc**2)

    vk = math.atan2(sqrt1_e2 * sinE, cosE - eph.ecc)
    phik = vk + eph.omega
    s2phi = math.sin(2.0 * phik)
    c2phi = math.cos(2.0 * phik)

    du = eph.cus * s2phi + eph.cuc * c2phi
    dr = eph.crs * s2phi + eph.crc * c2phi
    di = eph.cis * s2phi + eph.cic * c2phi

    u = phik + du
    r = A * (1 - eph.ecc * cosE) + dr
    i = eph.i0 + di + eph.idot * tk

    su = math.sin(u)
    cu = math.cos(u)
    si = math.sin(i)
    ci = math.cos(i)

    x_prime = r * cu
    y_prime = r * su

    if const == Constellation.BDS and (eph.prn <= 5 or eph.prn == 18 or eph.prn >= 59):
        omega_k = eph.omega0 + eph.omega_dot * tk - omega_dot_e * eph.toe.gps_timestamp
        sO = math.sin(omega_k)
        cO = math.cos(omega_k)
        xg = x_prime * cO - y_prime * ci * sO
        yg = x_prime * sO + y_prime * ci * cO
        zg = y_prime * si
        sinOe = math.sin(omega_dot_e * tk)
        cosOe = math.cos(omega_dot_e * tk)
        ca = math.cos(math.radians(-5.0))
        sa = math.sin(math.radians(-5.0))
        x = xg * cosOe + yg * sinOe * ca + zg * sinOe * sa
        y = -xg * sinOe + yg * cosOe * ca + zg * cosOe * sa
        z = -yg * sa + zg * ca
    else:
        omega_k = (
            eph.omega0 + (eph.omega_dot - omega_dot_e) * tk - omega_dot_e * eph.toe_sec
        )
        sO = math.sin(omega_k)
        cO = math.cos(omega_k)
        x = x_prime * cO - y_prime * ci * sO
        y = x_prime * sO + y_prime * ci * cO
        z = y_prime * si

    pos = np.array([x, y, z])
    return pos


def _gps_gal_bds_sat_info(
    t_sv: GpsTime,  # Signal pseudo transmission time
    eph: CdmaEphemeris,
    const: Constellation,
    mu: float = None,
    omega_e: float = None,
    F: float = None,
    group_delay_s: float = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute satellite information for GPS-like constellations."""

    tk = t_sv - eph.toe
    n, Ek = _compute_anomaly(tk, eph, mu)
    dt = t_sv - eph.toc
    clock_bias = (
        eph.sv_clock_bias
        + eph.sv_clock_drift * dt
        + eph.sv_clock_drift_rate * dt**2
        + F * eph.ecc * eph.sqrtA * math.sin(Ek)
        - group_delay_s
    )

    t_corr = t_sv.minus_float_seconds(clock_bias)
    tk = t_corr - eph.toe  # time from ephemeris reference epoch
    pos = _broadcast_orbit_pos(tk, eph, mu, omega_e, const)

    pos_fwd = _broadcast_orbit_pos(tk + 0.001, eph, mu, omega_e, const)
    vel = (pos_fwd - pos) / 0.001

    clock_drift = (
        eph.sv_clock_drift
        + 2 * eph.sv_clock_drift_rate * dt
        + n * F * eph.ecc * eph.sqrtA * math.cos(Ek) / (1 - eph.ecc * math.cos(Ek))
    )

    return (
        pos,
        vel,
        clock_bias * gnssConst.SPEED_OF_LIGHT_MS,
        group_delay_s * gnssConst.SPEED_OF_LIGHT_MS,
        clock_drift * gnssConst.SPEED_OF_LIGHT_MS,
    )


def _glo_sat_info(
    t_sv: GpsTime, eph: GloEphemeris
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    # TODO: Debug and validate GLONASS satellite position and clock computation
    tk = t_sv - eph.toc
    clock_bias = eph.sv_clock_bias + eph.sv_relative_freq_bias * tk
    t_corr = t_sv.gps_timestamp - clock_bias
    tk = t_corr - eph.toc.gps_timestamp

    state = np.array(
        [
            eph.x_pos,
            eph.y_pos,
            eph.z_pos,
            eph.x_vel,
            eph.y_vel,
            eph.z_vel,
        ]
    )
    acc = np.array([eph.x_acc, eph.y_acc, eph.z_acc])

    def _ode(y):
        x, y_, z, vx, vy, vz = y
        r = math.sqrt(x * x + y_ * y_ + z * z)
        mu = gnssConst.GloConstants.MU
        a_e = gnssConst.GloConstants.A_E
        c20 = gnssConst.GloConstants.C_20
        omega = gnssConst.GloConstants.OMEGA_DOT_E
        ax = (
            -mu * x / r**3
            - 1.5 * c20 * mu * (a_e**2 / r**5) * x * (1 - 5 * z * z / r**2)
            + omega**2 * x
            + 2 * omega * vy
            + acc[0]
        )
        ay = (
            -mu * y_ / r**3
            - 1.5 * c20 * mu * (a_e**2 / r**5) * y_ * (1 - 5 * z * z / r**2)
            + omega**2 * y_
            - 2 * omega * vx
            + acc[1]
        )
        az = (
            -mu * z / r**3
            - 1.5 * c20 * mu * (a_e**2 / r**5) * z * (3 - 5 * z * z / r**2)
            + acc[2]
        )
        return np.array([vx, vy, vz, ax, ay, az])

    h = 60.0 if tk >= 0 else -60.0
    t = 0.0
    steps = int(tk / h)
    rem = tk - steps * h
    y = state.copy()
    for _ in range(abs(steps)):
        k1 = _ode(y)
        k2 = _ode(y + 0.5 * h * k1)
        k3 = _ode(y + 0.5 * h * k2)
        k4 = _ode(y + h * k3)
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    if abs(rem) > 0:
        k1 = _ode(y)
        k2 = _ode(y + 0.5 * rem * k1)
        k3 = _ode(y + 0.5 * rem * k2)
        k4 = _ode(y + rem * k3)
        y += (rem / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    pos = y[:3]
    vel = y[3:]

    bias_plus = eph.sv_clock_bias + eph.sv_relative_freq_bias * (tk + 0.001)
    clock_drift = (bias_plus - clock_bias) / 0.001

    return (
        pos,
        vel,
        clock_bias * gnssConst.SPEED_OF_LIGHT_MS,
        0.0,
        clock_drift * gnssConst.SPEED_OF_LIGHT_MS,
    )


def compute_satellite_info(
    t_sv: GpsTime,  # Signal pseudo transmission time
    constellation: Constellation,
    eph: CdmaEphemeris | GloEphemeris,
    mu: float = None,
    omega_e: float = None,
    F_relativistic: float = None,
    group_delay_s: float = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """General dispatcher for satellite information computation."""

    if constellation in (Constellation.GPS, Constellation.GAL, Constellation.BDS):
        if (
            mu is None
            or omega_e is None
            or F_relativistic is None
            or group_delay_s is None
        ):
            raise ValueError(
                "mu, omega_e, and F must be provided for CDMA constellations"
            )
        return _gps_gal_bds_sat_info(
            t_sv, eph, constellation, mu, omega_e, F_relativistic, group_delay_s
        )
    if constellation == Constellation.GLO:
        return _glo_sat_info(t_sv, eph)
    raise ValueError("Unsupported constellation")
