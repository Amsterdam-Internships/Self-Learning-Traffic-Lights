import numpy as np
import matplotlib.pyplot as plt

import torch

from src.cityflow_env import CityFlowEnv
from src.dqn_agent import *
from src.utility import *
from src.evaluate import evaluate_one_traffic

"""
This file contains the training loop for the Deep Reinforcement Learning agent.
It constantly refreshes the replay memory by running the simulation engine
under the current epsilon-greedy policy of the trained agent. 

Source: https://medium.com/@unnatsingh/deep-q-network-with-pytorch-d1ca6f40bfda
"""

EPOCHS = 100
NUM_STEPS = 300
GOOD_REWARD = -1000
LR_START = 1e-4

args = parse_arguments()
config = update_config(NUM_STEPS)
intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
# state_size = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1
# current phase not necessary
state_size = len(config['lane_phase_info'][intersection_id]['start_lane'])
action_size = len(phase_list)
# to help him (otherwise he has to learn that using only these 2 actions is always better)
# TODO make it learn with all actions
action_size = 2
print("Action size = ", action_size)

env = CityFlowEnv(config)
agent = Agent(state_size, action_size, EPOCHS, seed=0)
state_normalizer = Normalizer(state_size)
reward_normalizer = Normalizer(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dqn(n_episodes=2, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
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
    lrs = []

    eps = eps_start
    lr = LR_START
    for epoch in range(1, n_episodes + 1):
        # training
        cumulative_loss, _ = run_env("train", eps)
        loss_episodes.append(cumulative_loss)

        # evaluation
        _, cumulative_reward = run_env("eval", 0)
        rewards_episodes.append(cumulative_reward)

        # if -3000 < cumulative_reward < -800:
        #     break

        eps = max(eps * eps_decay, eps_end)  # decrease  epsilon
        # lr = lr * 1/(1 + (LR_START/EPOCHS) * epoch)
        # for param_group in agent.optimizer.param_groups:
        #     param_group['lr' ] = lr

        agent.lr_scheduler.step()  # decrease learning rate
        lr = agent.lr_scheduler.get_last_lr()[0]
        lrs.append(lr)

        print('\rEpisode {}\tReward {:.2f}\tLoss {:.2f}\tLR: {:.2g}'.format(epoch, cumulative_reward, cumulative_loss,
                                                                            lr))

        # save model when good enough
        # average_size = 5
        # if len(rewards_episodes) > average_size and np.mean(rewards_episodes[:-average_size]) >= GOOD_REWARD:
        #     print('\nTrained in {:d} episodes.\tAverage of last {} cumulative rewards: {:.2f}\n'.format(epoch, average_size, np.mean(rewards_episodes[:-5])))
        #     torch.save(agent.qnetwork_local.state_dict(), 'trained_models/checkpoint.pth')
        #     break
        # torch.save(agent.qnetwork_local.state_dict(), 'trained_models/checkpoint.pth')

    return loss_episodes, rewards_episodes, lrs


def run_env(mode, eps):
    """Run 1 episode through environment.

    Params
    ======
        train (bool): training or evaluating
        eps (float): value of epsilon for epsilon-greedy action selection
    """

    state = env.reset()
    state = state_normalizer.normalize(state)

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

        # normalize state
        next_state = state_normalizer.normalize(next_state)

        if mode == "train":
            # normalize reward
            reward = reward_normalizer.normalize(np.array([reward]))

            # add to replay buffer and train
            agent.step(state, action - 1, reward, next_state, done)
            loss_episode += agent.loss

        if mode == "eval":
            cum_rewards += reward

        state = next_state
        last_action = action
        t += 1

    return loss_episode, cum_rewards


losses, rewards, learning_rates = dqn(EPOCHS)

# evaluate last run and make ready for cleaner visualisation
env.log()
evaluate_one_traffic(config, args.scenario, 'train', 'print')

# plot losses and rewards
fig = plt.figure()
skip = 5

ax = fig.add_subplot(221)
plt.plot(np.arange(skip, len(losses)), losses[skip:])
plt.ylabel('Loss')
plt.xlabel('Episode #')

fig.add_subplot(222)
plt.plot(np.arange(skip, len(rewards)), rewards[skip:])
plt.ylabel('Cumulative rewards')
plt.xlabel('After # episodes of learning')

fig.add_subplot(223)
plt.plot(np.arange(skip, len(learning_rates)), learning_rates[skip:])
plt.ylabel('Learning rate')
plt.xlabel('Episode')

fig.tight_layout()

save_plots("loss_and_reward")

# plt.show()
