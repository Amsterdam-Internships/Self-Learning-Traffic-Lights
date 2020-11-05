import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

import torch

from src.cityflow_env import CityFlowEnv
from src.dqn_agent import Agent
from src.utility import *
from src.evaluate import evaluate_one_traffic

"""
Source: https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
"""

# # add visible gpu if necessary
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

args = parse_arguments()
with open('src/config.json') as json_file:
    config = json.load(json_file)

config['num_step'] = 300
config['lane_phase_info'] = parse_roadnet("data/{}/roadnettest.json".format(args.scenario))
intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']

state_size = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1
action_size = len(phase_list)

env = CityFlowEnv(config)
agent = Agent(state_size, action_size, seed=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dqn(n_episodes=2, eps_start=1.0, eps_end=0.05, eps_decay=0.5):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    loss_episodes = []  # list containing cumulative loss per episode
    rewards_episodes = []  # list containing cumulative rewards per episode
    # scores_window = deque(maxlen=100)  # last 100 scores

    # evaluation before learning with random actions
    _, cumulative_reward = run_env(0, 1)
    rewards_episodes.append(cumulative_reward)

    eps = eps_start
    for i_episode in range(1, n_episodes + 1):

        # training
        cumulative_loss, _ = run_env(1, eps)
        loss_episodes.append(cumulative_loss)

        # evaluation
        _, cumulative_reward = run_env(0, 0)
        rewards_episodes.append(cumulative_reward)

        # scores_window.append(score)  # save the most recent score
        eps = max(eps * eps_decay, eps_end)  # decrease  epsilon

        print('\rEpisode {}\tAverage Reward {:.2f}'.format(i_episode, np.mean(rewards_episodes)))  # , end="")
        # print('\rEpisode {}\tAverage travel time {:.2f}'.format(i_episode, env.get_average_travel_time())) #  , end="")
        # if i_episode % 10 == 0:
        #     print('\rEpisode {}\tAverage Reward {:.2f}'.format(i_episode, np.mean(scores_window)))

        # if np.mean(scores_window) >= 200.0:
        #     print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode - 100,
        #                                                                                 np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #     break

    return loss_episodes, rewards_episodes


def run_env(train, eps):
    """Run 1 episode through environment

    Params
    ======
        train (bool): training or evaluating
        eps (float): value of epsilon for epsilon-greedy action selection
    """

    state = env.reset()

    loss_episode = 0
    cum_rewards = 0

    t = 0
    last_action = agent.act(state, eps) + 1  # phase_id starts from 1, yellow light is 0.
    while t < config['num_step']:
        action = agent.act(state, eps) + 1
        if action == last_action:
            next_state, reward, done, _ = env.step(action)
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
            next_state, reward, done, _ = env.step(action)

        if train:
            # phase_id starts from 1, yellow light is 0.
            # add to replay buffer and train
            agent.step(state, action - 1, reward, next_state, done)
            state = next_state

            loss_episode += agent.loss
        else:
            cum_rewards += reward

        last_action = action
        t += 1

    return loss_episode, cum_rewards


losses, rewards = dqn(10)

# should be in evaluation loop
# evaluate and make visualisation more cleanly possible
env.log()
# print(evaluate_one_traffic(config, args.scenario))


# plot losses and rewards
fig = plt.figure()
ax = fig.add_subplot(211)
skip = 0
plt.plot(np.arange(skip, len(losses)), losses[skip:])
plt.ylabel('Loss')
plt.xlabel('Episode #')

fig.add_subplot(212)
plt.plot(np.arange(len(rewards)), rewards)
plt.ylabel('Cumulative rewards')
plt.xlabel('After # episodes of learning')

fig.tight_layout()
save_plots("loss_and_reward")
plt.show()



