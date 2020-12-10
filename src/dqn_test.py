import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from src.cityflow_env import CityFlowEnv
from src.dqn_agent import Agent
from src.utility import *
from src.evaluate import evaluate_one_traffic

"""
This file contains the 

"""

args = parse_arguments()
NUM_STEPS = 300
NORM_INPUTS = 0  # Set to 1 to normalize inputs
NORM_REWARDS = 0  # Set to 1 to normalize rewards
TENSORBOARD = 0
config = setup_config(NUM_STEPS, 'test')

intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']

env = CityFlowEnv(config)
action_size = 2
# action_size = len(phase_list)
state_size = len(env.reset())
print(state_size)
agent = Agent(state_size, action_size, seed=0)
# load the weights from file
checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))
agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])

# add if statement and add saves to dqn_train
# self.state_normalizer = load_pickle("data/{}/state_normalizer".format(self.config['scenario']))
# self.reward_normalizer = load_pickle("data/{}/reward_normalizer".format(self.config['scenario']))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_dir = 'experiments/{}/test/tensorboard/'.format(args.exp_name) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if TENSORBOARD:
    writer = SummaryWriter(log_dir)


def run_env():
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}}
    state = env.reset()
    t = 0
    last_action = agent.act(state)
    while t < config['num_step']:
        action, q_values = agent.act(state)
        print(state)
        # Take step in environment, add yellow light if action changes
        if action == last_action:
            state, reward, done, _ = env.step(action)
        else:
            for _ in range(env.yellow_time):
                env.step(-1)  # action -1 -> yellow light
                stats['actions'][-1] += 1
                t += 1
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            state, reward, done, _ = env.step(action)

        last_action = action
        t += 1

        # Save evaluation stats
        stats['rewards'] += reward
        stats['actions'][action] += 1
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

