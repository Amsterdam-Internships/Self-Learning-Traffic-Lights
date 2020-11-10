import numpy as np
import matplotlib.pyplot as plt

import torch

from src.cityflow_env import CityFlowEnv
from src.dqn_agent import Agent
from src.utility import *
from src.evaluate import evaluate_one_traffic

"""
This file contains the 

"""

# TODO automatically set working directory

args = parse_arguments()
NUM_STEPS = 300
# todo make other flow files to test with (maybe he should open multiple?)
config = update_config(NUM_STEPS, 'test')

intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
state_size = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1
action_size = len(phase_list)
# to help him (otherwise he has to learn that using only these 2 actions is always better)
action_size = 2
print("Action size = ", action_size)

env = CityFlowEnv(config)
agent = Agent(state_size, action_size, seed=0)
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('trained_models/checkpoint.pth'))

# TODO necessary here?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# env is deterministic right?
def run_env():
    state = env.reset()
    t = 0
    last_action = agent.act(state) + 1  # phase_id starts from 1, yellow light is 0.
    while t < config['num_step']:
        action = agent.act(state) + 1
        if action == last_action:
            state, reward, done, _ = env.step(action)
        # if action changes, add a yellow light
        else:
            for _ in range(env.yellow_time):
                env.step(0)  # required yellow time
                t += 1
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            state, reward, done, _ = env.step(action)

        last_action = action
        t += 1


run_env()

env.log()
evaluate_one_traffic(config, args.scenario, 'test', 'print')
