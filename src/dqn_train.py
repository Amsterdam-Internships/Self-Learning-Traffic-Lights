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

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

args = parse_arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 5
NUM_STEPS = 300
TRAINING_RUNS = 3
NORM_TAU = 1e-3

NORM_INPUTS = 0  # Set to 1 to normalize inputs
NORM_REWARDS = 0  # Set to 1 to normalize rewards
LOAD = 0  # Set to 1 to load checkpoint
RANDOM_RUN = 0

if LOAD == 1:
    checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))

config = setup_config(NUM_STEPS, 'train', NORM_INPUTS, NORM_REWARDS, NORM_TAU)

intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
norm_state_size = len(config['lane_phase_info'][intersection_id]['start_lane'])

# action_size = 2
action_size = len(phase_list)

# The part of the state that is normalised + length one hot vector
state_size = norm_state_size + action_size

best_travel_time = 1000


def dqn(n_episodes, eps_start=0.9, eps_end=0.1):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
    """
    starting_epoch = 0
    eps = eps_start
    loss_episodes = []  # list containing cumulative loss per episode
    rewards_episodes = []  # list containing cumulative rewards per episode
    learning_rates = []
    epsilons = []
    mean_diff_q_vals = []
    mean_size_q_vals = []
    mean_diff_best_q_vals = []
    travel_times = []
    global best_travel_time

    agent = Agent(state_size, action_size, seed=0)

    # Load saved checkpoint
    if LOAD == 1:
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        # TODO load other stats

    for epoch in range(starting_epoch + 1, n_episodes + 1):
        # Perform training run through environment
        cumulative_loss, _, _, _, _, _, _, _ = run_env(agent, eps, "train")

        # Perform evaluation run through environment
        _, cumulative_reward, actions, diff_q_values, size_q_vals, diff_best_q_values, env, travel_time = run_env(agent, 0, "eval")

        # Decrease epsilon
        decay = (eps_start - eps_end) / ((n_episodes - starting_epoch) * 0.8)
        eps = max(eps - decay, eps_end)

        # Decrease learning rate
        agent.lr_scheduler.step()
        lr = agent.lr_scheduler.get_last_lr()[0]

        # Save best model
        if travel_time < best_travel_time:
            print('BEST\n')
            path = "trained_models/{}".format(args.exp_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.qnetwork_local.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, os.path.join(path, "checkpoint.tar"))
            env.log()
            best_travel_time = travel_time

        # Save stats
        loss_episodes.append(cumulative_loss)
        rewards_episodes.append(cumulative_reward)
        epsilons.append(round(eps, 5))
        learning_rates.append(lr)
        mean_diff_q_vals.append(np.round(np.mean(diff_q_values) ,3))
        mean_size_q_vals.append(np.round(np.mean(size_q_vals), 3))
        mean_diff_best_q_vals.append(np.round(np.mean(diff_best_q_values), 3))
        travel_times.append(travel_time)
        print('\rEpisode {}\tReward {}\tLoss {:.0f}\tLearning rate: {:.2g}\tEpsilon  {:.2g}\t Action count {}\t '
              'Mean difference in Q values {:.3g}\tTravel Time {}'.format(epoch, cumulative_reward, cumulative_loss, lr,
                                                                          eps,
                                                                          list(actions.values()),
                                                                          np.mean(diff_q_values), travel_time))

    return loss_episodes, rewards_episodes, learning_rates, epsilons, mean_diff_q_vals, mean_size_q_vals, mean_diff_best_q_vals, travel_times


def run_env(agent, eps, mode):
    """Run 1 episode through environment.

    Params
    ======
        agent (Agent): the DQN agent to train
        eps (float): value of epsilon for epsilon-greedy action selection
        mode (string): agent only takes step on 'train' mode
    """
    loss_episode = 0
    cum_rewards = 0
    actions = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    diff_q_values = []
    diff_best_q_values = []
    size_q_values = []
    t = 0

    env = CityFlowEnv(config)
    state = env.reset()
    last_action, _ = agent.act(state, eps)

    while t < config['num_step']:
        action, q_values = agent.act(state, eps)

        # Take step in environment, add yellow light if action changes
        if action == last_action:
            next_state, reward, done, _ = env.step(action)
        else:
            for _ in range(env.yellow_time):
                env.step(-1)  # action -1 -> yellow light
                actions[-1] += 1
                t += 1

                # Break out of the training loop when training steps is reached
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            next_state, reward, done, _ = env.step(action)

        # Add to replay buffer and train
        if mode == "train":
            agent.step(state, action, reward, next_state, done)
            loss_episode += agent.loss

        # Save stats
        actions[action] += 1
        cum_rewards += reward
        if q_values is not None:
            diff_q_values.append(np.round(np.mean([abs(q_values[0][action] - q_values[0][j]) for j in
                                                   range(len(q_values[0])) if
                                                   q_values[0][j] is not q_values[0][action]]), 3))
            size_q_values.append(np.round(np.mean([abs(q_values[0][j]) for j in range(len(q_values[0]))]), 3))
            q_values = np.sort(q_values[0])
            diff_best_q_values.append(round(abs(q_values[-1] - q_values[-2]), 3))

        state = next_state
        last_action = action
        t += 1

    return round(loss_episode, 2), round(cum_rewards, 2), actions, diff_q_values, size_q_values, diff_best_q_values, env, round(
        env.get_average_travel_time(), 2)


# Perform random run through environment before training
if RANDOM_RUN == 1:
    config["mode"] = 'random'
    _, cum_reward, _, _, _, _, environment, tt_random = run_env(Agent(state_size, action_size, seed=0), 1, "random")
    environment.log()  # TODO why should this be before config change? how can config change the environment?
    evaluate_one_traffic(config, args.scenario, 'random', 'print')
    print("reward and travel time when random, ", cum_reward, tt_random)
    config["mode"] = 'train'


# Take average over training runs
training_runs = []
for i in range(TRAINING_RUNS):
    training_runs.append(dqn(EPOCHS))
losses, rewards, lrs, epses, diff_q_vals, size_q_vals, diff_best_q_vals, tt = np.mean(training_runs, 0)
losses_std, rewards_std, _, _, diff_q_vals_std, size_q_vals_std, diff_best_q_vals_std, tt_std = np.std(training_runs, 0)

# Evaluate last run and make ready for cleaner visualisation
evaluate_one_traffic(config, args.scenario, 'train', 'print')

"""
PLOTS
"""

fig = plt.figure()
skip = 5

fig.add_subplot(331)
plt.plot(np.arange(skip, len(tt)), tt[skip:], color='red')
plt.fill_between(np.arange(skip, len(tt)), tt[skip:] - tt_std[skip:],
                 tt[skip:] + tt_std[skip:], facecolor='red',
                 alpha=0.2)
plt.ylabel('Travel Time')
plt.xlabel('Episodes')

fig.add_subplot(332)
plt.plot(np.arange(skip, len(rewards)), rewards[skip:], color='magenta')
plt.fill_between(np.arange(skip, len(rewards)), rewards[skip:] - rewards_std[skip:],
                 rewards[skip:] + rewards_std[skip:], facecolor='magenta',
                 alpha=0.2)
plt.ylabel('Cumulative rewards')
plt.xlabel('Episodes')

ax = fig.add_subplot(333)
plt.plot(np.arange(skip, len(losses)), losses[skip:], color='blue')
plt.fill_between(np.arange(skip, len(losses)), losses[skip:] - losses_std[skip:], losses[skip:] + losses_std[skip:],
                 facecolor='blue', alpha=0.2)
plt.ylabel('Loss')
plt.xlabel('Episode')

fig.add_subplot(334)
plt.plot(np.arange(skip, len(size_q_vals)), size_q_vals[skip:], color='brown')
plt.fill_between(np.arange(skip, len(size_q_vals)), size_q_vals[skip:] - size_q_vals_std[skip:],
                 size_q_vals[skip:] + size_q_vals_std[skip:], facecolor='brown',
                 alpha=0.2)
plt.ylabel('Q value size ')
plt.xlabel('Episodes')

fig.add_subplot(335)
plt.plot(np.arange(skip, len(diff_best_q_vals)), diff_best_q_vals[skip:], color='brown')
plt.fill_between(np.arange(skip, len(diff_best_q_vals)), diff_best_q_vals[skip:] - diff_best_q_vals_std[skip:],
                 diff_best_q_vals[skip:] + diff_best_q_vals_std[skip:], facecolor='brown',
                 alpha=0.2)
plt.ylabel('Difference best Q values')
plt.xlabel('Episodes')

fig.add_subplot(336)
plt.plot(np.arange(skip, len(diff_q_vals)), diff_q_vals[skip:], color='brown')
plt.fill_between(np.arange(skip, len(diff_q_vals)), diff_q_vals[skip:] - diff_q_vals_std[skip:],
                 diff_q_vals[skip:] + diff_q_vals_std[skip:], facecolor='brown',
                 alpha=0.2)
plt.ylabel('Difference all Q values')
plt.xlabel('Episodes')

fig.add_subplot(337)
plt.plot(np.arange(skip, len(lrs)), lrs[skip:], color='green')
plt.ylabel('Learning rate')
plt.xlabel('Episode')

fig.add_subplot(338)
plt.plot(np.arange(skip, len(epses)), epses[skip:], color='green')
plt.ylabel('Epsilon')
plt.xlabel('Episode')

fig.tight_layout()

save_plots("training_stats")

# plt.show()
