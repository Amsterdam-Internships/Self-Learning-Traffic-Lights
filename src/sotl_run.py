import json

from src.cityflow_env import CityFlowEnv
from src.utility import parse_roadnet
from src.utility import parse_arguments

"""
Source:https://github.com/tianrang-intelligence/TSCC2019
"""

with open('src/config.json') as json_file:
    config = json.load(json_file)

args = parse_arguments()
dataset = args.scenario
config['lane_phase_info'] = parse_roadnet("data/{}/roadnet.json".format(dataset))
config['num_step'] = 300

# TODO make argparse part of config
env = CityFlowEnv(config)

lane_phase_info = config['lane_phase_info']
intersection_id = list(lane_phase_info.keys())[0]
phase_list = lane_phase_info[intersection_id]["phase"]
phase_startLane_mapping = lane_phase_info[intersection_id]["phase_startLane_mapping"]
current_phase = phase_list[0]
current_phase_time = 0
yellow_time = 5
phase_log = []

phi = 20
min_green_vehicle = 2
max_red_vehicle = 5
action = phase_list[0]

intersection_id = list(lane_phase_info.keys())[0]
start_lane = lane_phase_info[intersection_id]['start_lane']


def choose_action(state):
    cur_phase = state["current_phase"]
    global action
    if state["current_phase_time"] >= phi:
        num_green_vehicle = sum(
            [state["lane_waiting_vehicle_count"][i] for i in phase_startLane_mapping[cur_phase]])
        num_red_vehicle = sum([state["lane_waiting_vehicle_count"][i] for i in
                               lane_phase_info[intersection_id]["start_lane"]]) - num_green_vehicle
        if num_green_vehicle <= min_green_vehicle and num_red_vehicle > max_red_vehicle:
            action = cur_phase % len(phase_list) + 1


def run_sotl():
    # reset initially
    t = 0
    env.reset()
    state = env.get_state_sotl()
    choose_action(state)
    last_action = action
    while t < config['num_step']:
        state = env.get_state_sotl()
        choose_action(state)
        if action == last_action:
            env.step(action)
        else:
            for _ in range(yellow_time):
                env.step(0)  # required yellow time
                t += 1
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            env.step(action)
        last_action = action
        t += 1


run_sotl()
print(env.get_average_travel_time())
