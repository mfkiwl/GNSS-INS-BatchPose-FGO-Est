import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fgo_solver.wls_solver import solve_wls_position
from gnss_utils.cycle_slip_detection import detect_cycle_slips
from gnss_utils.gnss_data_utils import apply_base_corrections, apply_ephemerides_to_obs
from gnss_utils.rinex_nav_parser import parse_rinex_nav
from gnss_utils.rinex_obs_parser import parse_rinex_obs
from gnss_utils.model_utils import compute_world_frame_coord_from_ecef
from imu_utils.imu_data_utils import parse_ground_truth_log


def _filter_epochs_with_gt(rover_obs, ground_truth_data):
    if not ground_truth_data:
        return rover_obs
    first_gt_epoch = min(ground_truth_data.keys())
    return {epoch: obs for epoch, obs in rover_obs.items() if epoch >= first_gt_epoch}


class TestWlsSolver(unittest.TestCase):
    def test_horizontal_accuracy_under_one_meter_for_ninety_percent(self):
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data" / "tex_cup"

        nav_file = data_dir / "brdm1290.19p"
        rover_file = data_dir / "asterx4_rover.obs"
        base_file = data_dir / "asterx4_base_1hz.obs"
        ground_truth_file = data_dir / "ground_truth.log"

        ground_truth_data = parse_ground_truth_log(str(ground_truth_file))

        eph_data = parse_rinex_nav(str(nav_file))
        rover_obs = parse_rinex_obs(str(rover_file))
        base_obs = parse_rinex_obs(str(base_file), interval=30)

        apply_ephemerides_to_obs(rover_obs, eph_data)
        eph_data.resetIndexLookup()
        apply_ephemerides_to_obs(base_obs, eph_data)
        apply_base_corrections(rover_obs, base_obs)
        detect_cycle_slips(rover_obs)

        rover_obs = _filter_epochs_with_gt(rover_obs, ground_truth_data)
        epochs_sorted = sorted(rover_obs.keys())
        epochs_subset = epochs_sorted[:200]

        horizontal_errors = []
        for epoch in epochs_subset:
            gt = ground_truth_data.get(epoch)
            if gt is None:
                continue

            channels = rover_obs.get(epoch)
            if not channels:
                continue

            try:
                est_ecef, _ = solve_wls_position(channels)
            except ValueError:
                continue

            est_enu = compute_world_frame_coord_from_ecef(est_ecef)
            diff = est_enu - gt.pos_world_enu_m
            horizontal_errors.append(float(np.linalg.norm(diff[:2])))

        self.assertTrue(
            horizontal_errors, "No valid WLS solutions were produced for evaluation."
        )

        errors = np.asarray(horizontal_errors)
        pct_within_one_meter = float(np.mean(errors < 1.0))

        # Print statistics (mean, std) for horizontal errors
        print(f"Mean horizontal error: {np.mean(errors):.3f} m")
        print(f"Standard deviation of horizontal errors: {np.std(errors):.3f} m")
        print(
            f"Percentage of horizontal errors below 1 meter: {pct_within_one_meter:.2%}"
        )

        self.assertGreater(
            pct_within_one_meter,
            0.99,
            f"Only {pct_within_one_meter:.2%} of horizontal errors are below 1 m",
        )


if __name__ == "__main__":
    unittest.main()
