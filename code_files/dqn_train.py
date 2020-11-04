import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch

from code_files.cityflow_env import CityFlowEnv
from code_files.dqn_agent import Agent
from code_files.utility import parse_roadnet
from code_files.utility import parse_arguments
from code_files.evaluate import evaluate_one_traffic

"""
Source: https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
"""

# # add visible gpu if necessary
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

args = parse_arguments()
with open('code_files/config.json') as json_file:
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


def dqn(n_episodes=2, num_step=config['num_step'], eps_start=1.0, eps_end=0.005, eps_decay=0.5):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

    """
    # lists containing scores of each episode
    scores_exploration = []
    # scores_exploitation = []
    scores_window = deque(maxlen=100)  # last 100 scores
    loss_episodes = []
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        loss_episode = 0

        t = 0
        last_action = agent.act(state, eps) + 1  # phase_id starts from 1, yellow light is 0.
        while t < num_step:
            action = agent.act(state, eps) + 1
            if action == last_action:
                next_state, reward, done, _ = env.step(action)
            # if action changes, add a yellow light
            else:
                for _ in range(env.yellow_time):
                    env.step(0)  # required yellow time
                    t += 1
                    flag = (t >= num_step)
                    if flag:
                        break
                if flag:  # TODO what happens here? why necessary?
                    break
                next_state, reward, done, _ = env.step(action)
            last_action = action
            t += 1

            agent.step(state, action - 1, reward, next_state, done)
            state = next_state

            score += reward
            loss_episode += agent.loss
            if done:
                break
            # print("episode: {}/{}, time: {}, action: {}, reward: {}"
            #       .format(i_episode, n_episodes, t - 1, action, reward))
            print('\rEpisode {}\tLast loss {:.2f}'.format(i_episode, agent.loss), end="")

        scores_window.append(score)  # save the most recent score
        scores_exploration.append(score)  # save the most recent score
        loss_episodes.append(loss_episode)
        eps = max(eps * eps_decay, eps_end)  # decrease  epsilon

        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)))  # , end="")
        # print('\rEpisode {}\tAverage travel time {:.2f}'.format(i_episode, env.get_average_travel_time())) #  , end="")
        # if i_episode % 10 == 0:
        #     print('\rEpisode {}\tAverage Reward {:.2f}'.format(i_episode, np.mean(scores_window)))

        # if np.mean(scores_window) >= 200.0:
        #     print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode - 100,
        #                                                                                 np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        #     break

    return scores_exploration, loss_episodes


scores, loss_episodes = dqn()

# evaluate and make visualisation more cleanly possible
env.log()
# print(evaluate_one_traffic(config, args.scenario))

# TODO make seperate function
eps = 0
state = env.reset()
score = []
cum = 0
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
    last_action = action
    t += 1
    cum += reward
    score.append(cum)

# plot rewards and losses
fig = plt.figure()
ax = fig.add_subplot(221)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Cumalative rewards')
plt.xlabel('Epsiode #')

fig.add_subplot(222)
plt.plot(np.arange(3, len(loss_episodes)), loss_episodes[3:])
plt.ylabel('Loss')
plt.xlabel('Epsiode #')

fig.add_subplot(223)
plt.plot(np.arange(len(score)), score)
plt.ylabel('Cumalative rewards')
plt.xlabel('Steps #')
plt.show()
