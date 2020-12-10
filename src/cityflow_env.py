import pandas as pd
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

        self.state_normalizer = Normalizer(len(config['lane_phase_info'][self.intersection_id]['start_lane']), config['norm_tau'])
        self.reward_normalizer = Normalizer(1, config['norm_tau'])

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
        lane_vehicle_count = [self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane]
        # phases = np.zeros(2)
        phases = np.zeros(len(self.phase_list))
        phases[self.current_phase] = 1

        if self.config['normalize_input'] == 1:
            self.state_normalizer.observe(np.array(lane_vehicle_count))
            lane_vehicle_count = self.state_normalizer.normalize(np.array(lane_vehicle_count))

        combined_state = np.array(list(lane_vehicle_count) + list(phases))

        return combined_state

    def get_state_sotl(self):
        state = {'lane_waiting_vehicle_count': self.eng.get_lane_waiting_vehicle_count(),
                 'current_phase': self.current_phase,
                 'current_phase_time': self.current_phase_time}
        return state

    def get_reward(self):
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        if self.config['normalize_rewards'] == 1:
            self.reward_normalizer.observe(np.array([reward]))
            reward = self.reward_normalizer.normalize(np.array([reward]))[0]
        return reward

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def log(self):
        """Saves chosen actions and normalizers to files.
        """
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

        # maybe path should still be created
        path = "experiments/{}/{}/{}".format(self.config['exp_name'], self.config["mode"], self.config['hyperparams'])
        df.to_csv(os.path.join(path, 'signal_plan_template.txt'), index=None)

        path = "trained_models/{}/{}".format(self.config["exp_name"], self.config['hyperparams'])
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)

        save_pickle(self.state_normalizer, os.path.join(path, "state_normalizer"))
        save_pickle(self.reward_normalizer, os.path.join(path, "reward_normalizer"))
