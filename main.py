import cityflow
from utility import parse_roadnet
import pandas as pd
import os
import json

with open('config.json') as json_file:
    config = json.load(json_file)

eng = cityflow.Engine("config.json", thread_num=1)
num_step = 1000

lane_phase_info = parse_roadnet("data/syn_1x1_uniform_200_1h/roadnet.json")
intersection_id = list(lane_phase_info.keys())[0]
phase_list = lane_phase_info[intersection_id]["phase"]
phase_startLane_mapping = lane_phase_info[intersection_id]["phase_startLane_mapping"]

phi = 20
min_green_vehicle = 2
max_red_vehicle = 5
action = phase_list[0]

intersection_id = list(lane_phase_info.keys())[0]
start_lane = lane_phase_info[intersection_id]['start_lane']

current_phase = phase_list[0]
current_phase_time = 0
yellow_time = 5

# reset initially
t = 0
eng.reset()
phase_log = []
state = {}


def get_state():
    state['lane_vehicle_count'] = eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
    state['start_lane_vehicle_count'] = {lane: eng.get_lane_vehicle_count()[lane] for lane in start_lane}
    state[
        'lane_waiting_vehicle_count'] = eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
    state['lane_vehicles'] = eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
    state['vehicle_speed'] = eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
    state['vehicle_distance'] = eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
    state['current_time'] = eng.get_current_time()
    state['current_phase'] = current_phase
    state['current_phase_time'] = current_phase_time


def choose_action():
    cur_phase = state["current_phase"]
    global action
    if state["current_phase_time"] >= phi:
        num_green_vehicle = sum(
            [state["lane_waiting_vehicle_count"][i] for i in phase_startLane_mapping[cur_phase]])
        num_red_vehicle = sum([state["lane_waiting_vehicle_count"][i] for i in
                               lane_phase_info[intersection_id]["start_lane"]]) - num_green_vehicle
        if num_green_vehicle <= min_green_vehicle and num_red_vehicle > max_red_vehicle:
            action = cur_phase % len(phase_list) + 1


def step(next_phase):
    global current_phase
    global current_phase_time
    if current_phase == next_phase:
        current_phase_time += 1
    else:
        current_phase = next_phase
        current_phase_time = 1

    eng.set_tl_phase(intersection_id, current_phase)
    eng.next_step()
    phase_log.append(current_phase)


get_state()
choose_action()
last_action = action
while t < num_step:
    get_state()
    choose_action()
    if action == last_action:
        step(action)
    else:
        for _ in range(yellow_time):
            step(0)  # required yellow time
            t += 1
            flag = (t >= num_step)
            if flag:
                break
        if flag:
            break
        step(action)
    last_action = action
    t += 1

df = pd.DataFrame({intersection_id: phase_log[:num_step]})
if not os.path.exists('data/syn_1x1_uniform_200_1h'):
    os.makedirs("data/syn_1x1_uniform_200_1h")
df.to_csv(os.path.join('data/syn_1x1_uniform_200_1h', 'signal_plan_template.txt'), index=None)
