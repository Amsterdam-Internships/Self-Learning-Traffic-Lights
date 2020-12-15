import json
import argparse
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

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


# save plots of experiment
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


def setup_config(data_set_mode, experiment_mode, lr=0, batch_size=0, norm_inputs=0, norm_rewards=0):
    """Update the configuration file

    Params
    ======
        num_steps (int): maximum number of training episodes
        mode (string):
        norm_inputs (bool):
        norm_rewards (bool):
        norm_tau (float):
    """
    args = parse_arguments()
    # update the config file with arguments
    with open('src/config.json') as json_file:
        config = json.load(json_file)

    config['hyperparams'] = "lr=" + str(lr) + "_batch_size=" + str(batch_size) + "_norm_rewards=" + str(norm_rewards) + "weight_norm"
    config["flowFile"] = "data/{}/{}/{}".format(args.scenario, data_set_mode, config["flowFile"])
    config["roadnetFile"] = "data/{}/{}/{}".format(args.scenario, data_set_mode, config["roadnetFile"])
    config['lane_phase_info'] = parse_roadnet(config["roadnetFile"])
    config["roadnetLogFile"] = "experiments/{}/{}/{}/{}".format(args.exp_name, experiment_mode, config['hyperparams'], config["roadnetLogFile"])
    config["replayLogFile"] = "experiments/{}/{}/{}/{}".format(args.exp_name, experiment_mode, config['hyperparams'], config["replayLogFile"])
    config['num_step'] = 300
    config['scenario'] = args.scenario
    config['mode'] = experiment_mode
    config['exp_name'] = args.exp_name
    config['normalize_input'] = norm_inputs
    config['normalize_rewards'] = norm_rewards
    config['norm_tau'] = 1e-3
    config['lr'] = lr
    config['batch_size'] = batch_size

    path = "experiments/{}".format(config['exp_name'])
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
    path = "experiments/{}/{}".format(config['exp_name'], config["mode"])
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    path = "experiments/{}/{}/{}".format(config['exp_name'], config["mode"], config['hyperparams'])
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    path = "trained_models/{}".format(args.exp_name)
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    path = "trained_models/{}/{}".format(args.exp_name, config['hyperparams'])
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)

    # write to file so the engine can open it.
    with open('src/config_args.json', 'w') as outfile:
        json.dump(config, outfile)

    return config


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """ Unpickle a file of pickled data. """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except IOError:
        pass


class Normalizer:
    def __init__(self, num_inputs, tau):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        self.tau = tau

    def observe(self, x):
        x = x.astype(np.float)
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean = (1 - self.tau) * self.mean + self.tau * x
        self.var = (1 - self.tau) * self.var + self.tau * (x - last_mean) * (x - self.mean)

    def normalize(self, x):
        eps = 1e-8
        return (x - self.mean)/(np.sqrt(self.var) + eps)
