import numpy as np
import constants.parameters as params
from fgo_solver.gnss_ins_fgo import RtkInsFgo
from gnss_utils.cycle_slip_detection import detect_cycle_slips
from gnss_utils.gnss_data_utils import apply_base_corrections, apply_ephemerides_to_obs
from gnss_utils.rinex_nav_parser import parse_rinex_nav
from gnss_utils.rinex_obs_parser import parse_rinex_obs
from imu_utils.imu_data_utils import parse_ground_truth_log, parse_imu_log


def _filter_epochs_with_gt(rover_obs, ground_truth_data):
    if not ground_truth_data:
        return rover_obs
    first_gt_epoch = min(ground_truth_data.keys())
    return {epoch: obs for epoch, obs in rover_obs.items() if epoch >= first_gt_epoch}


def main():
    nav_file = "data/tex_cup/brdm1290.19p"
    rover_file = "data/tex_cup/asterx4_rover.obs"
    base_file = "data/tex_cup/asterx4_base_1hz.obs"
    imu_file = "data/tex_cup/bosch_imu.log"
    ground_truth_file = "data/tex_cup/ground_truth.log"
    imu_params = params.TexCupBoschImuParams()

    ground_truth_data = {}
    if ground_truth_file is not None:
        ground_truth_data = parse_ground_truth_log(ground_truth_file)

    eph_data = parse_rinex_nav(nav_file)
    rover_obs = parse_rinex_obs(rover_file)
    base_obs = parse_rinex_obs(base_file, interval=30)
    imu_data_list = parse_imu_log(imu_file, imu_params.z_up)

    apply_ephemerides_to_obs(rover_obs, eph_data)
    eph_data.resetIndexLookup()
    apply_ephemerides_to_obs(base_obs, eph_data)
    apply_base_corrections(rover_obs, base_obs)
    detect_cycle_slips(rover_obs)

    rover_obs = _filter_epochs_with_gt(rover_obs, ground_truth_data)
    epochs_sorted = sorted(rover_obs.keys())
    if not epochs_sorted:
        print("No rover epochs available after ground truth alignment.")
        return

    start_idx = 200
    end_idx = min(900, len(epochs_sorted) - 1)
    if start_idx >= len(epochs_sorted):
        start_idx = 0

    solver = RtkInsFgo(
        imu_params,
        imu_data_list,
        window_size_s=5.0,
        show_progress=True,
    )

    results = solver.run(
        rover_obs,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    if not results:
        print("Solver produced no results.")
        return

    horizontal_errors = []
    vertical_errors = []
    cov_traces = []
    lever_arm_b = np.asarray(imu_params.t_imu_to_ant_in_b)

    for entry in results:
        gt = ground_truth_data.get(entry.epoch) if ground_truth_data else None
        if gt is None:
            continue
        rot_enu_from_body = entry.pose.rotation().matrix()
        lever_enu = rot_enu_from_body @ lever_arm_b
        est_ant_enu = entry.pose_enu_m + lever_enu
        diff_enu = est_ant_enu - gt.pos_world_enu_m
        horizontal_errors.append(float(np.linalg.norm(diff_enu[:2])))
        vertical_errors.append(float(diff_enu[2]))
        cov_traces.append(float(np.trace(entry.pose_cov_6x6)))

    if horizontal_errors:
        horizontal_errors = np.asarray(horizontal_errors)
        vertical_errors = np.asarray(vertical_errors)
        cov_traces = np.asarray(cov_traces)
        within_threshold = np.mean(horizontal_errors < 1.5) * 100.0
        print(f"Processed {len(results)} epochs in solver log.")
        print(
            f"Horizontal error < 1.5 m for {within_threshold:.1f}% of evaluated epochs"
        )
        print(f"Horizontal error RMS: {np.sqrt(np.mean(horizontal_errors**2)):.3f} m")
        print(f"Max horizontal error: {horizontal_errors.max():.3f} m")
        print(f"Vertical error RMS: {np.sqrt(np.mean(vertical_errors**2)):.3f} m")
        print(
            f"Mean pose covariance trace: {cov_traces.mean():.3e}, max: {cov_traces.max():.3e}"
        )
    else:
        print("Ground truth unavailable for evaluated epochs; skipping error metrics.")


if __name__ == "__main__":
    main()
