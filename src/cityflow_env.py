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

        path = "src/config_{}_args.json".format(config['scenario'])
        self.eng = cityflow.Engine(path, thread_num=1)

        self.config = config
        self.lane_phase_info = config['lane_phase_info']

        self.intersection_indices = config['intersection_indices']
        self.intersection_ids = list(self.lane_phase_info.keys())
        self.start_lanes = [self.lane_phase_info[intersection_id]['start_lane'] for intersection_id in list(self.lane_phase_info.keys())]
        self.end_lanes = [self.lane_phase_info[intersection_id]['end_lane'] for intersection_id in list(self.lane_phase_info.keys())]
        self.phase_lists = [self.lane_phase_info[intersection_id]['phase'] for intersection_id in list(self.lane_phase_info.keys())]
        self.phase_startLane_mappings = [self.lane_phase_info[intersection_id]['phase_startLane_mapping'] for intersection_id in list(self.lane_phase_info.keys())]
        self.phase_logs = [[] for intersection in range(len(self.intersection_ids))]
        self.current_phases = [0 for intersection in range(len(self.intersection_ids))]  # from -1 to len(phase_list)-1
        self.current_phase_times = [0 for intersection in range(len(self.intersection_ids))]

        # self.intersection_id = list(self.lane_phase_info.keys())[0]
        # self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        # self.end_lane = self.lane_phase_info[self.intersection_id]['end_lane']
        # self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        # self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]
        # self.current_phase = 0  # from -1 to len(phase_list)-1
        # self.current_phase_time = 0
        # self.phase_log = []

        self.last_phase = 0
        self.yellow_time = 5

        self.acyclic = config['acyclic']

        self.WAITING = config['waiting_added']
        self.DISTANCE = config['distance_added']
        self.SPEED = config['speed_added']

        self.ALL_VEHICLES_MAX = 40
        self.WAITING_MAX = 30
        self.MAX_DISTANCE = 300
        self.MAX_SPEED = 11

        # Not augmented for MA.
        self.last_reward = 0
        self.SHAPING_REWARD = 0
        self.TEMPORAL_REWARD = 0
        self.ADDED_REWARD = 0

        # self.state_normalizer = Normalizer(len(config['lane_phase_info'][self.intersection_id]['start_lane']), config['norm_tau'])
        # self.reward_normalizer = Normalizer(1, config['norm_tau'])

    def reset(self):
        self.eng.reset()
        self.phase_logs = [[] for phase in range(len(self.phase_lists))]
        return self.get_state()  # Only use when single agent.

    def step(self, next_phase, intersection_index=0):
        if self.acyclic:
            if self.current_phases[intersection_index] == next_phase:
                self.current_phase_times[intersection_index] += 1
            else:
                self.current_phases[intersection_index] = next_phase
                self.current_phase_times[intersection_index] = 1

            self.eng.set_tl_phase(self.intersection_ids[intersection_index], self.current_phases[intersection_index] + 1)  # +1 to make yellow light action 0.
            self.phase_logs[intersection_index].append(self.current_phases[intersection_index] + 1)
        else:
            if next_phase is not -1:
                if next_phase == 0:
                    self.current_phase_times[intersection_index] += 1
                if next_phase == 1:
                    self.current_phases[intersection_index] = (self.current_phases[intersection_index] + 1) % len(self.phase_lists[intersection_index])
                    self.current_phase_times[intersection_index] = 1

                self.eng.set_tl_phase(self.intersection_ids[intersection_index], self.current_phases[intersection_index] + 1)  # +1 to make yellow light action 0.
                self.phase_logs[intersection_index].append(self.current_phases[intersection_index] + 1)

            # Set yellow light.
            else:
                self.eng.set_tl_phase(self.intersection_ids[intersection_index], 0)
                self.phase_logs[intersection_index].append(0)

        # Takes a step in the environment only after all intersection-phase pairs are set.
        # Assuming the last intersection_id is used.
        if intersection_index == len(self.intersection_ids) - 1:
            self.eng.next_step()

        # Environment gives back the next_state and reward.
        return self.get_state(), self.get_reward()  # Only use when single agent.


    # # Only works with TIM method
    # def step_cyclic(self, switch):
    #     if switch is not -1:
    #         if switch == 0:
    #             self.current_phase_time += 1
    #         if switch == 1:
    #             if self.current_phase == len(self.phase_list) - 1:
    #                 self.current_phase = 0
    #             else:
    #                 self.current_phase = self.current_phase + 1
    #
    #         self.eng.set_tl_phase(self.intersection_id, self.current_phase + 1)  # +1 to make yellow light action 0.
    #         self.phase_log.append(self.current_phase + 1)
    #
    #     # Set yellow light.
    #     else:
    #         self.eng.set_tl_phase(self.intersection_id, 0)
    #         self.phase_log.append(0)
    #
    #     self.eng.next_step()
    #
    #     # Environment gives back the next_state and reward.
    #     return self.get_state(), self.get_reward()

    def get_state(self, intersection_index=0):

        lane_vehicle_count = [self.eng.get_lane_vehicle_count()[lane]/self.ALL_VEHICLES_MAX for lane in self.start_lanes[intersection_index]]

        # # Normalise LIT state.
        # if self.config['normalize_input'] == 1:
        #     self.state_normalizer.observe(np.array(lane_vehicle_count))
        #     lane_vehicle_count = self.state_normalizer.normalize(np.array(lane_vehicle_count))

        # Add current phase as a one-hot-vector.
        # CHANGE when straight
        # phases = np.zeros(2)
        if self.config['smdp'] == 1:
            phases = np.zeros(len(self.phase_lists[intersection_index]))
            if self.current_phases[intersection_index] is not -1:
                phases[self.current_phases[intersection_index]] = 1

        if self.config['smdp'] == 0:
            phases = np.zeros(len(self.phase_lists[intersection_index]) + 1)  # To represent yellow light as an additional phase.
            index = self.current_phases[intersection_index] + 1
            phases[index] = 1

        # State of LIT: all vehicles per lane + current phase.
        combined_state = lane_vehicle_count + list(phases)

        if self.WAITING:
            lane_waiting_vehicle_count = [self.eng.get_lane_waiting_vehicle_count()[lane]/self.WAITING_MAX for lane in self.start_lanes[intersection_index]]
            # TODO is this step unnecessary?
            lane_moving_vehicle_count = np.array(lane_vehicle_count) - np.array(lane_waiting_vehicle_count)

            # Moving and waiting vehicles per lane separated + current phase.
            combined_state = list(lane_moving_vehicle_count) + lane_waiting_vehicle_count + list(phases)

        # if self.DISTANCE:
        #     distance_per_vehicle = self.eng.get_vehicle_distance()
        #     print(distance_per_vehicle)
        #     # problem: distance travelled instead of distance from intersection.
        #
        #     combined_state = np.array(list(lane_vehicle_count) +
        #                               list(phases))
        if self.DISTANCE:
            distances_per_lane = [np.mean([float(self.eng.get_vehicle_info(vehicle_id)['distance'])
                                                             for vehicle_id in self.eng.get_lane_vehicles()[lane]])/self.MAX_DISTANCE
                                  if len(self.eng.get_lane_vehicles()[lane]) != 0 else 0.
                                  for lane in self.start_lanes[intersection_index]]
            combined_state = combined_state + distances_per_lane

        if self.SPEED:
            speeds_per_lane = [np.mean([float(self.eng.get_vehicle_info(vehicle_id)['speed'])
                                                             for vehicle_id in self.eng.get_lane_vehicles()[lane]])/self.MAX_SPEED
                                  if len(self.eng.get_lane_vehicles()[lane]) != 0 else 0.
                                  for lane in self.start_lanes[intersection_index]]
            combined_state = combined_state + speeds_per_lane

        # if self.SPEED:
        #     print('speed')
        #     speeds_per_lane = np.zeros(len(self.start_lane))
        #     vehicles_per_lane = np.zeros(len(self.start_lane))
        #
        #     speed_per_vehicle = self.eng.get_vehicle_speed()
        #     for vehicle_id, speed in speed_per_vehicle.items():
        #         for i, lane in enumerate(self.start_lane):
        #             vehicle_info = self.eng.get_vehicle_info(vehicle_id)
        #             if vehicle_info['drivable'] == lane:
        #                 if speed > 0.1:
        #                     speeds_per_lane[i] += speed
        #                     vehicles_per_lane[i] += 1
        #     average_speed_per_lane = np.nan_to_num(speeds_per_lane / vehicles_per_lane)
        #
        #     # Add average speed of moving cars per lane.
        #     combined_state = np.array(list(combined_state) +
        #                               list(average_speed_per_lane))

        # if self.DISTANCE:
        #     speeds_per_lane = np.zeros(len(self.start_lane))
        #     vehicles_per_lane = np.zeros(len(self.start_lane))
        #
        #     speed_per_vehicle = self.eng.get_vehicle_distance()
        #     for vehicle_id, speed in speed_per_vehicle.items():
        #         for i, lane in enumerate(self.start_lane):
        #             vehicle_info = self.eng.get_vehicle_info(vehicle_id)
        #             if vehicle_info['drivable'] == lane:
        #                 if speed > 0.1:
        #                     speeds_per_lane[i] += speed
        #                     vehicles_per_lane[i] += 1
        #     average_speed_per_lane = np.nan_to_num(speeds_per_lane / vehicles_per_lane)
        #
        #     # Add average speed of moving cars per lane.
        #     combined_state = np.array(list(combined_state) +
        #                               list(average_speed_per_lane))

        return combined_state

    def get_state_sotl(self):
        state = {'lane_waiting_vehicle_count': self.eng.get_lane_waiting_vehicle_count(),
                 'lane_vehicle_count': self.eng.get_lane_vehicle_count(),
                 'start_lane': self.start_lane,
                 'current_phase': self.current_phase,
                 'current_phase_time': self.current_phase_time}
        return state

    def get_reward(self, intersection_index=0):
        # lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        # reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        lane_waiting_vehicle_count = [self.eng.get_lane_waiting_vehicle_count()[lane] for lane in self.start_lanes[intersection_index]]
        reward = -1 * sum(lane_waiting_vehicle_count)
        if self.TEMPORAL_REWARD == 1:
            temp_reward = reward - self.last_reward
            self.last_reward = reward
            return temp_reward
        elif self.SHAPING_REWARD == 1:
            shape_reward = reward + self.last_reward
            self.last_reward = reward
            return shape_reward
        elif self.ADDED_REWARD == 1:
            temp_reward = reward - self.last_reward
            self.last_reward = reward
            return reward + temp_reward
        else:
            return reward

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def log(self):
        """Saves chosen actions and normalizers to files.
        """
        # args = parse_arguments()

        for intersection_index in self.intersection_indices:
            df = pd.DataFrame({self.intersection_ids[intersection_index]: self.phase_logs[intersection_index][:self.config['num_step']]})
            path = self.config['path_save']
            path = path + '/signal_plan_template' + str(intersection_index) + '.txt'
            df.to_csv(path, index=None)

        # path = "{}/trained_models/{}/{}".format(args.output_dir, self.config["exp_name"], self.config['hyperparams'])
        # if not os.path.exists(path):
        #     try:
        #         os.mkdir(path)
        #     except OSError:
        #         print("Creation of the directory %s failed" % path)
        #
        # save_pickle(self.state_normalizer, os.path.join(path, "state_normalizer"))
        # save_pickle(self.reward_normalizer, os.path.join(path, "reward_normalizer"))
