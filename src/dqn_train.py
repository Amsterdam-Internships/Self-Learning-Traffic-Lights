import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

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

EPOCHS = 1500
NUM_STEPS = 300
TRAINING_RUNS = 1
NORM_TAU = 1e-3
NORM_INPUTS = 0  # Set to 1 to normalize inputs
NORM_REWARDS = 0  # Set to 1 to normalize rewards
LOAD = 0  # Set to 1 to load checkpoint
RANDOM_RUN = 0

config = setup_config(NUM_STEPS, 'train', NORM_INPUTS, NORM_REWARDS, NORM_TAU)
intersection_id = list(config['lane_phase_info'].keys())[0]
phase_list = config['lane_phase_info'][intersection_id]['phase']
norm_state_size = len(config['lane_phase_info'][intersection_id]['start_lane'])
# action_size = 2
action_size = len(phase_list)
state_size = norm_state_size + action_size  # Normalised part + one-hot vector
best_travel_time = 100000

args = parse_arguments()
# om hyperparam search op te zetten: https://deeplizard.com/learn/video/ycxulUVoNbk
writer = SummaryWriter('experiments/{}/tensorboard_5'.format(args.exp_name), comment=f' batch_size={11} lr={0.1}')

# save network structure
writer.add_graph(Agent(state_size, action_size, 0).qnetwork_local,
                 torch.from_numpy(CityFlowEnv(config).reset()).unsqueeze(0).float())

if LOAD == 1:
    checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dqn(n_episodes, eps_start=0.9, eps_end=0.1):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
    """

    agent = Agent(state_size, action_size, 0)
    starting_epoch = 0
    eps = eps_start
    global best_travel_time

    # Load saved checkpoint
    if LOAD == 1:
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats = checkpoint['stats']
        # TODO load eps, maar werkt t wel om eps zo laag te laten bij trainen? misschien kan dit loaden en verder trainen niet echt.

    for epoch in range(starting_epoch + 1, n_episodes + 1):
        # Perform training run through environment
        train_stats, _ = run_env(agent, eps, "train")

        # Perform evaluation run through environment
        stats, env = run_env(agent, 0, "eval")
        stats['loss'] = train_stats['loss']

        # Decrease epsilon
        decay = (eps_start - eps_end) / ((n_episodes - starting_epoch) * 0.8)
        eps = max(eps - decay, eps_end)
        writer.add_scalar('Eps', eps, epoch)

        # Decrease learning rate
        agent.lr_scheduler.step()
        lr = agent.lr_scheduler.get_last_lr()[0]
        writer.add_scalar('LR', lr, epoch)

        # Save best model
        if stats['travel_time'] < best_travel_time:
            print('BEST\n')
            path = "trained_models/{}".format(args.exp_name)
            torch.save({
                'stats': stats,
                'model_state_dict': agent.qnetwork_local.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, os.path.join(path, "checkpoint.tar"))
            env.log()
            best_travel_time = stats['travel_time']

        print_every = 1
        if epoch % print_every == print_every - 1:
            print('\rEpisode {}\tMean Reward{:.2f}\tLoss {:.0f}\tLearning rate: {:.2g}\tEpsilon  {:.2g}\t Action count {}'
                  '\tTravel Time {:.0f}\tQ value size {:.0f}'.format(epoch, stats['rewards']/(NUM_STEPS - stats['actions'][0]), stats['loss'], lr,
                                                eps,
                                                list(stats['actions'].values()),
                                                stats['travel_time'], np.mean(stats['q_values_size'])))

        for name, weight in agent.qnetwork_local.named_parameters():
            writer.add_histogram(name + '_qnetwork_local', weight, epoch)
        for name, weight in agent.qnetwork_target.named_parameters():
            writer.add_histogram(name + '_qnetwork_target', weight, epoch)


def run_env(agent, eps, mode):
    """Run 1 episode through environment.

    Params
    ======
        agent (Agent): the DQN agent to train
        eps (float): value of epsilon for epsilon-greedy action selection
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'loss': 0, 'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
             'travel_time': 0, 'q_values_size': []}

    # diff_q_values = []
    # diff_best_q_values = []
    # size_q_values = []

    t = 0
    env = CityFlowEnv(config)
    state = env.reset()
    last_action, _ = agent.act(state, eps)

    while t < config['num_step']:
        # todo return mid activations of layers too and viz
        action, q_values = agent.act(state, eps)

        # Take step in environment, add yellow light if action changes
        if action == last_action:
            next_state, reward, done, _ = env.step(action)
        else:
            for _ in range(env.yellow_time):
                env.step(-1)  # action -1 -> yellow light
                stats['actions'][-1] += 1
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
            writer.add_scalar('Loss', agent.loss, agent.training_step)
            stats['loss'] += agent.loss
            agent.training_step += 1

        # Save stats
        if mode == "eval":
            writer.add_scalar('Reward', reward, agent.acting_step)
            writer.add_histogram('Actions', action, agent.acting_step)
            writer.add_histogram('Q values', q_values, agent.acting_step)
            stats['actions'][action] += 1
            stats['rewards'] += reward
            agent.acting_step += 1

        # if q_values is not None:
        #     diff_q_values.append(np.round(np.mean([abs(q_values[0][action] - q_values[0][j]) for j in
        #                                            range(len(q_values[0])) if
        #                                            q_values[0][j] is not q_values[0][action]]), 3))
            stats['q_values_size'].append(np.round(np.mean([abs(q_values[0][j]) for j in range(len(q_values[0]))]), 3))
        #     q_values = np.sort(q_values[0])
        #     diff_best_q_values.append(round(abs(q_values[-1] - q_values[-2]), 3))

        state = next_state
        last_action = action
        t += 1

    stats['travel_time'] = env.get_average_travel_time()
    return stats, env


# Perform random run through environment before training
if RANDOM_RUN == 1:
    config["mode"] = 'random'
    _, environment = run_env(Agent(state_size, action_size, seed=0), 1, "random")
    environment.log()  # TODO why should this be before config change? how can config change the environment?
    evaluate_one_traffic(config, args.scenario, 'random', 'print')
    # print("reward and travel time when random, ", cum_reward, tt_random)
    config["mode"] = 'train'

# Take average over training runs
training_runs = []
# als je deze gemiddeldes wil blijven bewaren moet je alles pas hierna naar de writer schrijven, en wat meer returnen.
# of dit checken https://doneill612.github.io/Scalar-Summaries/
for i in range(TRAINING_RUNS):
    training_runs.append(dqn(EPOCHS))

# Evaluate last run and make ready for cleaner visualisation
evaluate_one_traffic(config, args.scenario, 'train', 'print')

# make sure that all pending events have been written to disk.
writer.flush()
writer.close()
