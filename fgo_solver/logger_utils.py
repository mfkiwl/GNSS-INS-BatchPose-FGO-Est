"""Structured logging utilities for debugging solver epochs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import gtsam

from gnss_utils.time_utils import GpsTime


@dataclass
class MeasurementDebugRecord:
    constellation: str
    prn: int
    signal_type: str
    code_residual: Optional[float] = None
    phase_residual: Optional[float] = None
    code_used: bool = False
    phase_used: bool = False
    rejected: bool = False
    ambiguity_forced: bool = False
    prior_ambiguity: Optional[float] = None
    note: Optional[str] = None


def _fmt(value: Optional[float], fmt: str = "{:.4f}") -> str:
    return fmt.format(value) if value is not None else "-"


def log_epoch_debug(
    logger: logging.Logger,
    epoch: GpsTime,
    predicted_pose: gtsam.Pose3,
    predicted_velocity: Optional[np.ndarray],
    pose_std: Optional[np.ndarray],
    vel_std: Optional[np.ndarray],
    measurements: Sequence[MeasurementDebugRecord],
    *,
    skipped: bool = False,
) -> None:
    """Emit detailed debug information for a specific solver epoch."""

    if logger is None or not logger.isEnabledFor(logging.INFO):
        return

    pos = predicted_pose.translation()
    yaw, pitch, roll = predicted_pose.rotation().ypr()
    euler_deg = np.degrees([yaw, pitch, roll])

    if pose_std is not None and pose_std.shape[0] >= 6:
        rot_std_deg = np.degrees(pose_std[:3])
        pos_std = pose_std[3:]
    else:
        rot_std_deg = np.full(3, np.nan)
        pos_std = np.full(3, np.nan)

    if predicted_velocity is None:
        vel = np.full(3, np.nan)
    else:
        vel = np.asarray(predicted_velocity, dtype=float)

    if vel_std is None or vel_std.shape[0] < 3:
        vel_std_vec = np.full(3, np.nan)
    else:
        vel_std_vec = vel_std[:3]

    epoch_str = epoch.toDatetimeInUtc().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    logger.info(
        "Epoch %s%s | pos=[%.3f %.3f %.3f] m (std=[%s %s %s]) "
        "vel=[%.3f %.3f %.3f] m/s (std=[%s %s %s]) "
        "euler=[%.3f %.3f %.3f] deg (std=[%s %s %s])",
        epoch_str,
        " (GNSS skipped)" if skipped else "",
        pos[0],
        pos[1],
        pos[2],
        _fmt(pos_std[0]),
        _fmt(pos_std[1]),
        _fmt(pos_std[2]),
        vel[0],
        vel[1],
        vel[2],
        _fmt(vel_std_vec[0]),
        _fmt(vel_std_vec[1]),
        _fmt(vel_std_vec[2]),
        euler_deg[0],
        euler_deg[1],
        euler_deg[2],
        _fmt(rot_std_deg[0]),
        _fmt(rot_std_deg[1]),
        _fmt(rot_std_deg[2]),
    )

    for rec in measurements:
        logger.info(
            "    %s PRN %02d %s | code_res=%s m (used=%s) "
            "phase_res=%s m (used=%s) rejected=%s amb_forced=%s prior_N=%s %s",
            rec.constellation,
            rec.prn,
            rec.signal_type,
            _fmt(rec.code_residual),
            rec.code_used,
            _fmt(rec.phase_residual),
            rec.phase_used,
            rec.rejected,
            rec.ambiguity_forced,
            _fmt(rec.prior_ambiguity),
            f"[{rec.note}]" if rec.note else "",
        )

