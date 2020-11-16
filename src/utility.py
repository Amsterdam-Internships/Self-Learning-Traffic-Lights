import json
import argparse
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

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


def update_config(num_steps, mode='train'):
    args = parse_arguments()

    # update the config file with arguments
    with open('src/config.json') as json_file:
        config = json.load(json_file)

    config["dir"] = "data/{}/{}/".format(args.scenario, mode)

    roadnet = config["roadnetFile"]
    config['lane_phase_info'] = parse_roadnet(os.path.join(config["dir"], roadnet))
    config['num_step'] = num_steps
    config['scenario'] = args.scenario
    config['mode'] = mode
    config['exp_name'] = args.exp_name

    # write to file so the engine can open it.
    with open('src/config_args.json', 'w') as outfile:
        json.dump(config, outfile)

    return config


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print('saved', filename)


def load_pickle(filename):
    """ Unpickle a file of pickled data. """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except IOError:
        pass


def init_normalizer(epochs, num_steps, env, action_size, norm_state_size):
    args = parse_arguments()
    state_normalizer = Normalizer(norm_state_size)
    reward_normalizer = Normalizer(1)

    for _ in range(epochs):
        env.reset()
        t = 0
        last_action = random.choice(np.arange(action_size))
        while t < num_steps:
            action = random.choice(np.arange(action_size))
            if action == last_action:
                next_state, reward, _, _ = env.step(action)
            # if action changes, add a yellow light
            else:
                for _ in range(env.yellow_time):
                    env.step(-1)  # required yellow time
                    t += 1
                next_state, reward, _, _ = env.step(action)

            state_normalizer.observe(next_state[:norm_state_size])
            reward_normalizer.observe(np.array([reward]))
            last_action = action
            t += 1

    save_pickle(state_normalizer, "data/{}/state_normalizer".format(args.scenario))
    save_pickle(reward_normalizer, "data/{}/reward_normalizer".format(args.scenario))


class Normalizer:
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        x = x.astype(np.float)
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.var += ((x - last_mean) * (x - self.mean) - self.var) / self.n

    def normalize(self, x):
        eps = 1e-8
        return (x - self.mean)/(np.sqrt(self.var) + eps)
