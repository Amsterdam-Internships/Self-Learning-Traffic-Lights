import json
import argparse
import os
import matplotlib.pyplot as plt

"""
This file contains various helper methods.

Source: https://github.com/tianrang-intelligence/TSCC2019
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="new_scenario")
    parser.add_argument("--num_step", type=int, default=300)
    parser.add_argument("--exp_name", type=str, default="new_experiment")
    return parser.parse_args()


def parse_roadnet(roadnet_file):
    roadnet = json.load(open(roadnet_file))
    lane_phase_info_dict = {}

    # many intersections exist in the roadnet and virtual intersection is controlled by signal
    for intersection in roadnet["intersections"]:
        if intersection['virtual']:
            continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                    "end_lane": [],
                                                    "phase": [],
                                                    "phase_startLane_mapping": {},
                                                    "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]

        start_lane = []
        end_lane = []
        # roadLink includes some lane_pair: (start_lane, end_lane)
        road_link_lane_pair = {ri: [] for ri in range(len(road_links))}

        for ri in range(len(road_links)):
            road_link = road_links[ri]
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                start_lane.append(sl)
                end_lane.append(el)
                road_link_lane_pair[ri].append((sl, el))

        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))

        for phase_i in range(1, len(intersection["trafficLight"]["lightphases"])):
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            for ri in p["availableRoadLinks"]:
                lane_pair.extend(road_link_lane_pair[ri])
                if road_link_lane_pair[ri][0][0] not in start_lane:
                    start_lane.append(road_link_lane_pair[ri][0][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']]["phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][phase_i] = lane_pair

    return lane_phase_info_dict


def save_plots(name):
    args = parse_arguments()
    # define the name of the directory to be created
    path = os.path.join("experiments/", args.exp_name)

    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    os.chdir(path)
    plt.savefig(name)
    os.chdir("../")


def update_config(num_steps, mode='train'):
    args = parse_arguments()

    # update the config file with arguments
    with open('src/config.json') as json_file:
        config = json.load(json_file)

    config["dir"] = "data/{}/{}/".format(args.scenario, mode)

    roadnet = config["roadnetFile"]
    config['lane_phase_info'] = parse_roadnet(os.path.join(config["dir"], roadnet))
    config['num_step'] = num_steps

    # write to file so the engine can open it.
    with open('src/config_args.json', 'w') as outfile:
        json.dump(config, outfile)

    return config
