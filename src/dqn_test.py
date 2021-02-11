import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from src.cityflow_env import CityFlowEnv
from src.dqn_agent import Agent
from src.dqn_train import *
from src.utility import *
from src.evaluate import evaluate_one_traffic

"""
This file contains the test loop for the Deep Reinforcement Learning agent.
ONZIN FILE

"""

args = parse_arguments()
TENSORBOARD = 0

config = setup_config(NUM_STEPS, 'test')

intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']

env = CityFlowEnv(config)
action_size = len(phase_list)
state_size = len(env.reset())
print(state_size)
agent = Agent(state_size, action_size, seed=0)

# load the weights from file
checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))
agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        if TENSORBOARD:
            writer.add_scalar('Reward', reward, t)
            writer.add_scalar('Q value 1', q_values[0][0], t)

        print('\rReward {:.2f}\tQ value_1 {:.2f}\tAction {}'.format(reward, q_values[0][0], action))

    print('\rMean Reward {:.2f}\nActions on test flow {}'.format(stats['rewards']/(config['num_step'] - stats['actions'][-1]), list(stats['actions'].values())))


run_env()

evaluation_train = 'experiments/{}/train/evaluation.txt'.format(args.exp_name)
f = open(evaluation_train, "r")
print('Actions on train flow: ', f.read())

env.log()
evaluate_one_traffic(config, args.scenario, 'test', 'print')

