import cityflow
import pandas as pd
import os


class CityFlowEnv:
    """
    Simulator Environment with CityFlow
    """
    def __init__(self, config):
        self.eng = cityflow.Engine("config.json", thread_num=1)

        self.config = config
        self.lane_phase_info = config['lane_phase_info']

        self.intersection_id = list(self.lane_phase_info.keys())[0]
        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        self.yellow_time = 5

        self.phase_log = []

    def reset(self):
        self.eng.reset()
        self.phase_log = []

    def step(self, next_phase):
        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        self.eng.next_step()
        self.phase_log.append(self.current_phase)

    def get_state(self):
        state = {'lane_vehicle_count': self.eng.get_lane_vehicle_count(),
                 'start_lane_vehicle_count': {lane: self.eng.get_lane_vehicle_count()[lane] for lane in
                                              self.start_lane},
                 'lane_waiting_vehicle_count': self.eng.get_lane_waiting_vehicle_count(),
                 'lane_vehicles': self.eng.get_lane_vehicles(), 'vehicle_speed': self.eng.get_vehicle_speed(),
                 'vehicle_distance': self.eng.get_vehicle_distance(), 'current_time': self.eng.get_current_time(),
                 'current_phase': self.current_phase, 'current_phase_time': self.current_phase_time}

        return state

    def get_reward(self):
        # a sample reward function which calculates the total of waiting vehicles
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        return reward

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def get_score(self):
        tt = self.eng.get_average_travel_time()
        b = 100
        score = (b - tt) / b
        return score

    def log(self):
        # self.eng.print_log(self.config['replay_data_path'] + "/replay_roadnet.json",
        #                    self.config['replay_data_path'] + "/replay_flow.json")
        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.config['num_step']]})
        if not os.path.exists(self.config['dir']):
            os.makedirs(self.config['dir'])
        df.to_csv(os.path.join(self.config['dir'], 'signal_plan_template.txt'), index=None)
