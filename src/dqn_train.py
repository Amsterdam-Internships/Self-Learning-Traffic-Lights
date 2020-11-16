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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
NUM_STEPS = 300
TRAINING_RUNS = 3

args = parse_arguments()
config = update_config(NUM_STEPS)
# set to 1 if normalizer should be initialized
config['init_normalizer'] = 0
# set to 1 to normalize
config['normalize_input'] = 1
config['normalize_rewards'] = 1
load = 0

intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
norm_state_size = len(config['lane_phase_info'][intersection_id]['start_lane'])
# the part of the state that is normalised + length one hot vectors
state_size = norm_state_size + 1
# action_size = len(phase_list)
action_size = 2

if config['init_normalizer'] == 1:
    init_normalizer(10, NUM_STEPS, CityFlowEnv(config), action_size, norm_state_size)
    config['init_normalizer'] = 0

reward_normalizer = load_pickle("data/{}/reward_normalizer".format(args.scenario))
state_normalizer = load_pickle("data/{}/state_normalizer".format(args.scenario))
print("reward mean en variance = ", reward_normalizer.mean, reward_normalizer.var)
print("state mean en variance= ", state_normalizer.mean, state_normalizer.var)

best_cum_reward = -10000


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
    starting_epoch = 1

    if load == 1:
        checkpoint = torch.load("trained_models/checkpoint")
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    loss_episodes = []  # list containing cumulative loss per episode
    rewards_episodes = []  # list containing cumulative rewards per episode
    learning_rates = []
    epsilons = []
    mean_diff_q_vals = []
    global best_cum_reward

    eps = eps_start
    for epoch in range(starting_epoch, n_episodes + 1):
        # training
        cumulative_loss, _, _, _, _ = run_env("train", eps, agent)

        # evaluation
        _, cumulative_reward, actions, diff_q_values, env = run_env("eval", 0, agent)

        decay = (eps_start - eps_end) / (n_episodes * 0.8)
        eps = max(eps - decay, eps_end)

        agent.lr_scheduler.step()  # decrease learning rate
        lr = agent.lr_scheduler.get_last_lr()[0]

        loss_episodes.append(cumulative_loss)
        rewards_episodes.append(cumulative_reward)
        epsilons.append(round(eps, 5))
        learning_rates.append(lr)
        mean_diff_q_vals.append(np.mean(diff_q_values))

        print('\rEpisode {}\tReward {}\tLoss {:.0f}\tLearning rate: {:.2g}\tEpsilon  {:.2g}\t Action count {}\t '
              'Mean difference in Q values {:.3g}'.format(epoch, cumulative_reward, cumulative_loss, lr, eps,
                                                          list(actions.values()), np.mean(diff_q_values)))
        if cumulative_reward > best_cum_reward:
            best_cum_reward = cumulative_reward
            print('BEST\n')
            env.log()
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.qnetwork_local.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, 'trained_models/checkpoint.tar')

    return loss_episodes, rewards_episodes, learning_rates, epsilons, mean_diff_q_vals


def run_env(mode, eps, agent):
    """Run 1 episode through environment.

    Params
    ======
        train (bool): training or evaluating
        eps (float): value of epsilon for epsilon-greedy action selection
    """
    env = CityFlowEnv(config)
    state = env.reset()

    loss_episode = 0
    cum_rewards = 0
    actions = {0: 0, 1: 0}
    diff_q_values = []
    t = 0
    last_action, _ = agent.act(state, eps)
    while t < config['num_step']:
        action, q_values = agent.act(state, eps)
        if q_values is not None:
            diff_q_values.append(round(abs(q_values[0][0] - q_values[0][1]), 3))
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
            cum_rewards += reward

        state = next_state
        last_action = action
        t += 1
    return round(loss_episode, 2), cum_rewards, actions, diff_q_values, env


# Average over training runs
training_runs = []
for i in range(TRAINING_RUNS):
    training_runs.append(dqn(EPOCHS))
losses, rewards, lrs, epses, diff_q_vals = np.mean(training_runs, 0)
losses_std, rewards_std, _, _, diff_q_vals_std = np.std(training_runs, 0)

# evaluate last run and make ready for cleaner visualisation
evaluate_one_traffic(config, args.scenario, 'train', 'print')

"""
PLOTS
"""

fig = plt.figure()
skip = 5

ax = fig.add_subplot(321)
plt.plot(np.arange(skip, len(losses)), losses[skip:], color='blue')
plt.fill_between(np.arange(skip, len(losses)), losses[skip:] - losses_std[skip:], losses[skip:] + losses_std[skip:],
                 facecolor='blue', alpha=0.2)
plt.ylabel('Loss')
plt.xlabel('Episode')

fig.add_subplot(322)
plt.plot(np.arange(skip, len(rewards)), rewards[skip:], color='red')
plt.fill_between(np.arange(skip, len(rewards)), rewards[skip:] - rewards_std[skip:],
                 rewards[skip:] + rewards_std[skip:], facecolor='red',
                 alpha=0.2)
plt.ylabel('Cumulative rewards')
plt.xlabel('Episodes')

fig.add_subplot(323)
plt.plot(np.arange(skip, len(lrs)), lrs[skip:], color='green')
plt.ylabel('Learning rate')
plt.xlabel('Episode')

fig.add_subplot(324)
plt.plot(np.arange(skip, len(epses)), epses[skip:], color='green')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

fig.add_subplot(325)
plt.plot(np.arange(skip, len(diff_q_vals)), diff_q_vals[skip:], color='magenta')
plt.fill_between(np.arange(skip, len(diff_q_vals)), diff_q_vals[skip:] - diff_q_vals_std[skip:],
                 diff_q_vals[skip:] + diff_q_vals_std[skip:], facecolor='magenta',
                 alpha=0.2)
plt.ylabel('Mean difference in Q values')
plt.xlabel('Episodes')

fig.tight_layout()

save_plots("training_stats")

# plt.show()
