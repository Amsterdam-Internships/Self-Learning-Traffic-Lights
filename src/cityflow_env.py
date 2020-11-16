import pandas as pd
import os
import numpy as np
import json
from src.utility import *

import cityflow

"""
This file sets up the CityFlow simulation engine based on the config.json file.

Source: https://github.com/tianrang-intelligence/TSCC2019
"""


class CityFlowEnv:
    """
    Simulator Environment
    """
    def __init__(self, config):

        self.eng = cityflow.Engine("src/config_args.json", thread_num=1)

        self.config = config
        self.lane_phase_info = config['lane_phase_info']

        self.intersection_id = list(self.lane_phase_info.keys())[0]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.end_lane = self.lane_phase_info[self.intersection_id]['end_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        self.yellow_time = 5

        self.phase_log = []

        self.state_normalizer = load_pickle("data/{}/state_normalizer".format(self.config['scenario']))
        self.reward_normalizer = load_pickle("data/{}/reward_normalizer".format(self.config['scenario']))

    def reset(self):
        self.eng.reset()
        self.phase_log = []
        return self.get_state()

    def step(self, next_phase):
        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1
        self.eng.set_tl_phase(self.intersection_id, self.current_phase + 1)  # +1 to make yellow light action 0.
        self.phase_log.append(self.current_phase + 1)
        self.eng.next_step()

        # environment gives back: next_state, reward, done, _
        return self.get_state(), self.get_reward(), 0, 'niks'

    def get_state(self):
        state = {'start_lane_vehicle_count': {lane: self.eng.get_lane_vehicle_count()[lane] for lane in
                                              self.start_lane},
                 'current_phase': self.current_phase}
        if self.config['init_normalizer'] == 1 or self.config['normalize_input'] == 0:
            state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']])
        else:
            norm_state = self.state_normalizer.normalize(np.array(list(state['start_lane_vehicle_count'].values())))
            state = np.array(list(norm_state) + [state['current_phase']])

        # TODO add one-hot of actions
        return state

    def get_state_sotl(self):
        state = {'lane_waiting_vehicle_count': self.eng.get_lane_waiting_vehicle_count(),
                 'current_phase': self.current_phase,
                 'current_phase_time': self.current_phase_time}
        return state

    def get_reward(self):
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        if self.config['init_normalizer'] == 0 and self.config['normalize_rewards'] == 1:
            self.reward_normalizer.normalize(np.array([reward]))

        # all_vehicles_speeds = self.eng.get_vehicle_speed()
        # reward = np.amin(list(all_vehicles_speeds.values()))

        # reward = 0
        # if self.current_phase > 7:
        #     reward = 10
        return reward

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def log(self):
        # self.eng.print_log(self.config['replay_data_path'] + "/replay_roadnet.json",
        #                    self.config['replay_data_path'] + "/replay_flow.json")
        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.config['num_step']]})
        path = "experiments/{}".format(self.config['exp_name'])
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
        path = "experiments/{}/{}".format(self.config['exp_name'], self.config["mode"])
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
        df.to_csv(os.path.join(path, 'signal_plan_template.txt'), index=None)
