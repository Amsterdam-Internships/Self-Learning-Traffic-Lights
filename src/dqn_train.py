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

EPOCHS = 150
NUM_STEPS = 300
GOOD_REWARD = -1000

args = parse_arguments()
config = update_config(NUM_STEPS)
intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
norm_state_size = len(config['lane_phase_info'][intersection_id]['start_lane'])
# the part of the state that is normalised + length one hot vectors
state_size = norm_state_size + 1
# action_size = len(phase_list)
action_size = 2

# fill_normalizer(100, 100, CityFlowEnv(config), action_size, norm_state_size)

env = CityFlowEnv(config)

print('normalized')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dqn(n_episodes=2, eps_start=0.9, eps_end=0.1, eps_decay=0.995):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    agent = Agent(state_size, action_size, seed=0)

    loss_episodes = []  # list containing cumulative loss per episode
    rewards_episodes = []  # list containing cumulative rewards per episode
    learning_rates = []
    epsilons = []

    eps = eps_start
    for epoch in range(1, n_episodes + 1):
        # training
        cumulative_loss, _ = run_env("train", eps, agent)
        loss_episodes.append(cumulative_loss)

        # evaluation
        _, cumulative_reward = run_env("eval", 0, agent)
        rewards_episodes.append(cumulative_reward)

        decay = (eps_start - eps_end) / (n_episodes * 0.8)
        eps = max(eps - decay, eps_end)
        epsilons.append(eps)

        agent.lr_scheduler.step()  # decrease learning rate
        lr = agent.lr_scheduler.get_last_lr()[0]
        learning_rates.append(lr)

        print('\rEpisode {}\tReward {:.2f}\tLoss {:.2f}\tLearning rate: {:.2g}\tEpsilon  {:.2g}'.format(epoch, cumulative_reward,
                                                                                    cumulative_loss,
                                                                                    lr, eps))

        # save model when good enough
        # average_size = 5
        # if len(rewards_episodes) > average_size and np.mean(rewards_episodes[:-average_size]) >= GOOD_REWARD:
        #     print('\nTrained in {:d} episodes.\tAverage of last {} cumulative rewards: {:.2f}\n'.format(epoch, average_size, np.mean(rewards_episodes[:-5])))
        #     torch.save(agent.qnetwork_local.state_dict(), 'trained_models/checkpoint.pth')
        #     break
        # torch.save(agent.qnetwork_local.state_dict(), 'trained_models/checkpoint.pth')

    return loss_episodes, rewards_episodes, learning_rates, epsilons


def run_env(mode, eps, agent):
    """Run 1 episode through environment.

    Params
    ======
        train (bool): training or evaluating
        eps (float): value of epsilon for epsilon-greedy action selection
    """

    state = env.reset()

    loss_episode = 0
    cum_rewards = 0
    actions = {0:0, 1:0, 2:0}
    t = 0
    last_action = agent.act(state, eps)
    while t < config['num_step']:
        action = agent.act(state, eps)
        if action == last_action:
            next_state, reward, done, _ = env.step(action)
        # if action changes, add a yellow light
        else:
            for _ in range(env.yellow_time):
                env.step(-1)  # required yellow time
                t += 1
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            next_state, reward, done, _ = env.step(action)

        if mode == "train":
            # add to replay buffer and train
            agent.step(state, action, reward, next_state, done)
            loss_episode += agent.loss

        if mode == "eval":
            actions[action] += 1
            if action == 1 and cum_rewards < 0:
                x=2
            cum_rewards += reward

        state = next_state
        last_action = action
        t += 1
    # print(actions)
    return loss_episode, cum_rewards


# Average over training runs
training_runs = []
for i in range(3):
    training_runs.append(dqn(EPOCHS))
losses, rewards, lrs, epses = np.mean(training_runs, 0)
losses_std, rewards_std, _, _ = np.std(training_runs, 0)

# run 1 training loop
# losses, rewards, lrs, epses = dqn(EPOCHS)

# evaluate last run and make ready for cleaner visualisation
env.log()
evaluate_one_traffic(config, args.scenario, 'train', 'print')

"""
PLOTS
"""

fig = plt.figure()
skip = 5

ax = fig.add_subplot(221)
plt.plot(np.arange(skip, len(losses)), losses[skip:], color='blue')
plt.fill_between(np.arange(skip, len(losses)), losses[skip:] - losses_std[skip:], losses[skip:] + losses_std[skip:], facecolor='blue', alpha=0.2)
plt.ylabel('Loss')
plt.xlabel('Episode')

fig.add_subplot(222)
plt.plot(np.arange(skip, len(rewards)), rewards[skip:], color='red')
plt.fill_between(np.arange(skip, len(rewards)), rewards[skip:] - rewards_std[skip:], rewards[skip:] + rewards_std[skip:], facecolor='red',
                 alpha=0.2)
plt.ylabel('Cumulative rewards')
plt.xlabel('Episodes')

fig.add_subplot(223)
plt.plot(np.arange(skip, len(lrs)), lrs[skip:], color='green')
plt.ylabel('Learning rate')
plt.xlabel('Episode')

fig.add_subplot(224)
plt.plot(np.arange(skip, len(epses)), epses[skip:], color='green')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

fig.tight_layout()

save_plots("training_stats")

# plt.show()
