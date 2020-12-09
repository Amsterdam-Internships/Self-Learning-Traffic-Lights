import json
import os

from src.cityflow_env import CityFlowEnv
from src.utility import *
from src.evaluate import *

"""
This file contains the rule-based Self-Organising-Traffic-Lights algorithm to be used as a baseline.

Source:https://github.com/tianrang-intelligence/TSCC2019
"""

args = parse_arguments()
config = setup_config('train', 'sotl')

env = CityFlowEnv(config)

lane_phase_info = config['lane_phase_info']
intersection_id = list(lane_phase_info.keys())[0]
phase_list = lane_phase_info[intersection_id]["phase"]
phase_startLane_mapping = lane_phase_info[intersection_id]["phase_startLane_mapping"]
# current_phase = phase_list[0]
# current_phase_time = 0
yellow_time = 5
# phase_log = []

phi = 20
min_green_vehicle = 10 # this doesnt have any effect for some reason (maybe because of the 1 of max_red)
max_red_vehicle = 1
action = phase_list[0]

intersection_id = list(lane_phase_info.keys())[0]
start_lane = lane_phase_info[intersection_id]['start_lane']


def choose_action(state):
    cur_phase = state["current_phase"]
    global action
    if state["current_phase_time"] >= phi:
        num_green_vehicle = sum(
            [state["lane_waiting_vehicle_count"][i] for i in phase_startLane_mapping[cur_phase+1]])
        num_red_vehicle = sum([state["lane_waiting_vehicle_count"][i] for i in
                               lane_phase_info[intersection_id]["start_lane"]]) - num_green_vehicle
        if num_green_vehicle <= min_green_vehicle and num_red_vehicle > max_red_vehicle:
            # if cur_phase == len(phase_list)-1:
            if cur_phase == 1:  # use if only 2 actions (because it goes through full action cycle)
                action = 0
            else:
                action += 1


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
                env.step(-1)  # required yellow time
                t += 1
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            env.step(action)
        last_action = action
        t += 1
    env.log()
    evaluate_one_traffic(config, args.scenario, 'sotl', 'print')
