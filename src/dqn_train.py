import datetime
import linecache
import tracemalloc

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


TENSORBOARD = 1
TIM = 1
LOAD = 0  # Set to 1 to load checkpoint
EPS_START = 0.9
EPS_END = 0.1

args = parse_arguments()


def dqn(n_trajactories, config):
    """ Deep Q-Learning

    Params
    ======
        n_trajactories (int): maximum number of training episodes
        config (json): configuration file to setup the CityFlow engine
    """

    env = CityFlowEnv(config)
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    # CHANGE when straight
    # action_size = 2
    action_size = len(phase_list)
    state_size = len(env.reset())
    best_travel_time = 100000
    starting_trajectory = 0
    eps = EPS_START

    agent = Agent(state_size, action_size, 0, config['lr'], config['batch_size'])

    if TENSORBOARD:
        log_dir = 'experiments/{}/tensorboard/{}'.format(args.exp_name, config[
            'hyperparams']) + "_time=" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir, comment=f' batch_size={11} lr={0.1}')

    # Load saved checkpoint (because of epsilon not in use).
    if LOAD == 1:
        checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats = checkpoint['stats']

    for trajectory in range(starting_trajectory + 1, n_trajactories + 1):

        # Perform training run through environment.
        if TIM:
            run_env_tim(agent, eps, config, env, "train", trajectory)
        else:
            run_env(agent, eps, config, env, "train", trajectory)

        # Decrease epsilon.
        decay = (EPS_START - EPS_END) / ((n_trajactories - starting_trajectory) * 0.8)
        eps = max(eps - decay, EPS_END)

        # Decrease learning rate.
        agent.lr_scheduler.step()
        lr = agent.lr_scheduler.get_last_lr()[0]

        # Save and show training stats.
        stats_every = 10
        if trajectory % stats_every == stats_every - 1:

            # Perform evaluation run through environment.
            if TIM:
                stats = run_env_tim(agent, 0, config, env, "eval", trajectory)
            else:
                stats = run_env(agent, 0, config, env, "eval", trajectory)
            print(
                '\rTrajactory {}\tTravel Time {:.0f}\tMean Reward{:.2f}\tBatch_size {}\tLearning rate: {:.2g}\tEpsilon '
                '{:.2g}\t Action count {}'.format(trajectory + 1, stats['travel_time'],
                                                  stats['rewards'] / config['num_step'],
                                                  config['batch_size'], lr,
                                                  eps,
                                                  list(stats['actions'].values())))
            # Save best model.
            if stats['travel_time'] < best_travel_time:
                print('BEST\n')
                path = "trained_models/{}/{}".format(args.exp_name, config['hyperparams'])
                torch.save({
                    'stats': stats,
                    'model_state_dict': agent.qnetwork_local.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, os.path.join(path, "checkpoint.tar"))
                env.log()
                best_travel_time = stats['travel_time']

            if TENSORBOARD:
                writer.add_scalar('Eps', eps, trajectory)
                writer.add_scalar('LR', lr, trajectory)
                for name, weight in agent.qnetwork_local.named_parameters():
                    writer.add_histogram(name + '_qnetwork_local', weight, trajectory)
                for name, weight in agent.qnetwork_target.named_parameters():
                    writer.add_histogram(name + '_qnetwork_target', weight, trajectory)

                writer.add_scalar('Average Reward', stats['rewards'] / config['num_step'], trajectory)
                writer.add_scalar('Average Travel Time', stats['travel_time'], trajectory)

    # Make sure that all pending events have been written to disk.
    if TENSORBOARD:
        writer.flush()
        writer.close()

    # Make the replay logs represent the best trajectory instead of the last trajectory.
    evaluate_one_traffic(config, args.scenario, 'train', 'print')


def run_env(agent, eps, config, env, mode=None, epoch=0):
    """Run 1 episode through environment.

    Params
    ======
        agent (Agent): the DQN agent to train
        eps (float): value of epsilon for epsilon-greedy action selection
        config (json): configuration file to setup the CityFlow engine
        env (CityFlowEnv): CityFlow environment
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
             'travel_time': 0}

    t = 0
    state = env.reset()
    last_action = agent.act(state, eps)

    while t < config['num_step']:
        action = agent.act(state, eps)

        # Take step in environment, add yellow light if action changes.
        if action == last_action:
            next_state, reward = env.step(action)
        else:
            reward = 0
            for _ in range(env.yellow_time):
                _, sub_reward = env.step(-1)  # action -1 -> yellow light
                # TODO add discount factor. so subreward * discount^counter, dat is in de reward. maar moet de discount factor
                # voor de q value ook tot de macht 5?
                reward += sub_reward
                stats['actions'][-1] += 1
                t += 1

                # Break out of the training loop when training steps is reached.
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            next_state, sub_reward = env.step(action)
            reward += sub_reward

        # Add to replay buffer and train.
        if mode == "train":
            agent.step(state, action, reward, next_state)

        # Save evaluation stats.
        if mode == "eval":
            stats['actions'][action] += 1
            stats['rewards'] += reward

        state = next_state
        last_action = action
        t += 1

    stats['travel_time'] = env.get_average_travel_time()
    return stats


def run_env_tim(agent, eps, config, env, mode=None, epoch=0):
    """Run 1 episode through environment.

    Params
    ======
        agent (Agent): the DQN agent to train
        eps (float): value of epsilon for epsilon-greedy action selection
        config (json): configuration file to setup the CityFlow engine
        env (CityFlowEnv): CityFlow environment
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
             'travel_time': 0}

    yellow_time = 0
    state = env.reset()
    last_action = agent.act(state, eps)

    for t in range(config['num_step']):

        action = agent.act(state, eps)

        if yellow_time == 0:
            if action == last_action:
                next_state, reward = env.step(action)
            else:
                next_state, reward = env.step(-1)
                yellow_time = 1

        elif 0 < yellow_time < env.yellow_time:
            next_state, reward = env.step(-1)
            yellow_time += 1

        elif yellow_time == env.yellow_time:
            next_state, reward = env.step(action)
            yellow_time = 0

        # Add to replay buffer and train.
        if mode == "train":
            agent.step(state, action, reward, next_state)

        # Save evaluation stats.
        if mode == "eval":
            stats['actions'][action] += 1
            stats['rewards'] += reward

        state = next_state
        last_action = action

    stats['travel_time'] = env.get_average_travel_time()
    return stats


def random_run():
    """ Perform random run through environment without training
    """

    config = setup_config('train', 'random')
    env = CityFlowEnv(config)

    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    action_size = len(phase_list)
    state_size = len(env.reset())
    random_agent = Agent(state_size, action_size, seed=0)

    run_env(random_agent, 1, config, env, 'random')
    env.log()
    evaluate_one_traffic(config, args.scenario, 'random', 'print')
