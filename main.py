from utilities.rinex_nav_parser import parse_rinex_nav
from utilities.rinex_obs_parser import parse_rinex_obs
from utilities.gnss_data_utils import apply_ephemerides_to_obs
from constants.gnss_constants import Constellation


if __name__ == "__main__":
    nav_file = "data/tex_cup/brdm1290.19p"
    eph_data = parse_rinex_nav(nav_file)

    rover_file = "data/tex_cup/asterx4_rover.obs"
    base_file = "data/tex_cup/asterx4_base_1hz.obs"

    rover_obs = parse_rinex_obs(rover_file)
    base_obs = parse_rinex_obs(base_file, interval=30)

    apply_ephemerides_to_obs(rover_obs, eph_data)
    eph_data.resetIndexLookup()
    apply_ephemerides_to_obs(base_obs, eph_data)

    # Example usage: query GPS PRN 1 ephemeris at first epoch
    query_time = eph_data.gps_ephemerides[1][0][0]
    eph = eph_data.getCurrentEphemeris(Constellation.GPS, 1, query_time)
    print("Example ephemeris for GPS PRN 1 at", query_time, ":")
    print(eph.__dict__)

    print(
        f"Loaded {len(rover_obs)} rover epochs and {len(base_obs)} base epochs after applying ephemerides"
    )
