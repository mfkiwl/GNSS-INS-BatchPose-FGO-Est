"""Utility functions for satellite position, velocity and clock computations."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

import constants.gnss_constants as gnssConst
from utilities.gnss_data_structures import (
    Constellation,
    EphemerisData,
    GpsEphemeris,
    GloEphemeris,
    GalEphemeris,
    BdsEphemeris,
    GnssMeasurementChannel,
    SignalChannelId,
)
from utilities.time_utils import GpsTime


_C = gnssConst.SPEED_OF_LIGHT_MS


def _kepler_anomaly(ecc: float, mean_anom: float) -> float:
    """Solve Kepler's equation for eccentric anomaly."""
    E = mean_anom
    for _ in range(30):
        dE = (E - ecc * math.sin(E) - mean_anom) / (1 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < 1e-13:
            break
    return E


def _broadcast_orbit_pos(
    tk: float,
    eph,
    mu: float,
    omega_dot_e: float,
    const: Constellation,
) -> Tuple[np.ndarray, float]:
    """Return satellite position and eccentric anomaly."""

    A = eph.sqrtA ** 2
    n0 = math.sqrt(mu / A ** 3)
    n = n0 + eph.delta_n
    Mk = eph.m0 + n * tk
    Ek = _kepler_anomaly(eph.ecc, Mk)
    sinE = math.sin(Ek)
    cosE = math.cos(Ek)
    sqrt1_e2 = math.sqrt(1.0 - eph.ecc ** 2)

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

    if const == Constellation.BDS and (
        eph.prn <= 5 or eph.prn == 18 or eph.prn >= 59
    ):
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
        omega_k = eph.omega0 + (eph.omega_dot - omega_dot_e) * tk - omega_dot_e * eph.toe.gps_timestamp
        sO = math.sin(omega_k)
        cO = math.cos(omega_k)
        x = x_prime * cO - y_prime * ci * sO
        y = x_prime * sO + y_prime * ci * cO
        z = y_prime * si

    pos = np.array([x, y, z])
    return pos, Ek


def _gps_gal_bds_sat_info(time: GpsTime, eph, const: Constellation) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute satellite information for GPS-like constellations."""

    if const == Constellation.GPS:
        mu = gnssConst.GpsConstants.MU
        omega_e = gnssConst.GpsConstants.OMEGA_DOT_E
        F = gnssConst.GpsConstants.F
        group_delay = getattr(eph, "tgd", 0.0)
    elif const == Constellation.GAL:
        mu = gnssConst.GalConstants.MU
        omega_e = gnssConst.GalConstants.OMEGA_DOT_E
        F = gnssConst.GalConstants.F
        # Use E1/E5a bias if available else E1/E5b
        group_delay = eph.bgd_e1e5a if not math.isnan(eph.bgd_e1e5a) else eph.bgd_e1e5b
    elif const == Constellation.BDS:
        mu = gnssConst.BdsConstants.MU
        omega_e = gnssConst.GpsConstants.OMEGA_DOT_E
        F = gnssConst.BdsConstants.F
        group_delay = getattr(eph, "tgd1", 0.0)
    else:
        raise ValueError("Invalid constellation for this routine")

    dt_sv = 0.0
    for _ in range(2):
        t = time.gps_timestamp - dt_sv
        tk = t - eph.toe.gps_timestamp
        pos, Ek = _broadcast_orbit_pos(tk, eph, mu, omega_e, const)
        dt = t - eph.toc.gps_timestamp
        dt_sv = (
            eph.sv_clock_bias
            + eph.sv_clock_drift * dt
            + eph.sv_clock_drift_rate * dt ** 2
            + F * eph.ecc * eph.sqrtA * math.sin(Ek)
        )

    t_corr = time.gps_timestamp - dt_sv
    tk = t_corr - eph.toe.gps_timestamp
    pos, Ek = _broadcast_orbit_pos(tk, eph, mu, omega_e, const)

    dt = t_corr - eph.toc.gps_timestamp
    clock_bias = (
        eph.sv_clock_bias
        + eph.sv_clock_drift * dt
        + eph.sv_clock_drift_rate * dt ** 2
        + F * eph.ecc * eph.sqrtA * math.sin(Ek)
    )

    h = 0.001
    pos_fwd, _ = _broadcast_orbit_pos(tk + h, eph, mu, omega_e, const)
    pos_back, _ = _broadcast_orbit_pos(tk - h, eph, mu, omega_e, const)
    vel = (pos_fwd - pos_back) / (2 * h)

    dt_sv_plus = 0.0
    t_plus = time.gps_timestamp + h
    for _ in range(2):
        t_p = t_plus - dt_sv_plus
        tk_p = t_p - eph.toe.gps_timestamp
        _, Ek_p = _broadcast_orbit_pos(tk_p, eph, mu, omega_e, const)
        dt = t_p - eph.toc.gps_timestamp
        dt_sv_plus = (
            eph.sv_clock_bias
            + eph.sv_clock_drift * dt
            + eph.sv_clock_drift_rate * dt ** 2
            + F * eph.ecc * eph.sqrtA * math.sin(Ek_p)
        )

    clock_drift = (dt_sv_plus - clock_bias) / h

    return pos, vel, clock_bias * _C, group_delay * _C, clock_drift * _C


def _glo_sat_info(time: GpsTime, eph: GloEphemeris) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    tk = time.gps_timestamp - eph.toc.gps_timestamp
    clock_bias = eph.sv_clock_bias + eph.sv_relative_freq_bias * tk
    t_corr = time.gps_timestamp - clock_bias
    tk = t_corr - eph.toc.gps_timestamp

    state = np.array([
        eph.x_pos,
        eph.y_pos,
        eph.z_pos,
        eph.x_vel,
        eph.y_vel,
        eph.z_vel,
    ])
    acc = np.array([eph.x_acc, eph.y_acc, eph.z_acc])

    def _ode(y):
        x, y_, z, vx, vy, vz = y
        r = math.sqrt(x * x + y_ * y_ + z * z)
        mu = gnssConst.GloConstants.MU
        a_e = gnssConst.GloConstants.A_E
        c20 = gnssConst.GloConstants.C_20
        omega = gnssConst.GloConstants.OMEGA_DOT_E
        ax = (
            -mu * x / r ** 3
            - 1.5 * c20 * mu * (a_e ** 2 / r ** 5) * x * (1 - 5 * z * z / r ** 2)
            + omega ** 2 * x
            + 2 * omega * vy
            + acc[0]
        )
        ay = (
            -mu * y_ / r ** 3
            - 1.5 * c20 * mu * (a_e ** 2 / r ** 5) * y_ * (1 - 5 * z * z / r ** 2)
            + omega ** 2 * y_
            - 2 * omega * vx
            + acc[1]
        )
        az = (
            -mu * z / r ** 3
            - 1.5 * c20 * mu * (a_e ** 2 / r ** 5) * z * (3 - 5 * z * z / r ** 2)
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

    return pos, vel, clock_bias * _C, 0.0, clock_drift * _C


def compute_satellite_info(
    time: GpsTime,
    constellation: Constellation,
    eph,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """General dispatcher for satellite information computation."""

    if constellation in (Constellation.GPS, Constellation.GAL, Constellation.BDS):
        return _gps_gal_bds_sat_info(time, eph, constellation)
    if constellation == Constellation.GLO:
        return _glo_sat_info(time, eph)
    raise ValueError("Unsupported constellation")


def apply_ephemerides_to_obs(
    obs: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]],
    eph_data: EphemerisData,
) -> None:
    """Populate satellite information for all observation channels.

    Channels without valid ephemeris are removed from ``obs``.
    """

    for epoch in list(obs.keys()):
        channels = obs[epoch]
        to_delete = []
        for ch_id, ch in channels.items():
            eph = eph_data.getCurrentEphemeris(
                ch_id.signal_type.constellation, ch_id.prn, epoch
            )
            if eph is None:
                to_delete.append(ch_id)
                continue
            ch.computeSatelliteInformation(eph)
            if ch.sat_pos_ecef_m is None:
                to_delete.append(ch_id)
        for d in to_delete:
            channels.pop(d, None)
        if not channels:
            obs.pop(epoch)

