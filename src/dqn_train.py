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

TENSORBOARD = 0
LOAD = 0  # Set to 1 to load checkpoint
EPS_START = 1
EPS_END = 0.1
EPS_END_PERCENTAGE = 0.1
GAMMA = 0.99  # discount factor (Should be same as in agent file)

args = parse_arguments()


def dqn(n_trajactories, time, lr, batch_size, rm_size, learn_every, smdp, waiting_added, distance_added, speed_added):
    """ Deep Q-Learning

    Params
    ======
        n_trajactories (int): maximum number of training episodes
        config (json): configuration file to setup the CityFlow engine
    """
    config_val = setup_config(args.scenario_val, 'val', time, lr, batch_size, rm_size, learn_every, smdp, waiting_added,
                 distance_added, speed_added)
    env_val = CityFlowEnv(config_val)

    intersection_id = list(config_val['lane_phase_info'].keys())[0]
    phase_list = config_val['lane_phase_info'][intersection_id]['phase']
    action_size = len(phase_list)
    state_size = len(env_val.reset())
    best_travel_time_train = np.ones(len(args.scenarios_train)) * 1000000
    best_travel_time_val = 100000
    starting_trajectory = 0
    eps = EPS_START
    data_set_index = 0

    agent = Agent(state_size, action_size, 0, lr, batch_size, rm_size,
                  learn_every)

    if TENSORBOARD:
        log_dir = '{}/experiments/{}/tensorboard/{}'.format(args.output_dir, args.exp_name, config_val[
            'hyperparams']) + "_time=" + time
        writer = SummaryWriter(log_dir, comment=f' batch_size={11} lr={0.1}')

    for trajectory in range(starting_trajectory + 1, n_trajactories + 1):
        # TODO make list of configs at the start, and list of envs.
        config = setup_config(args.scenarios_train[data_set_index], 'train', time, lr, batch_size, rm_size, learn_every, smdp, waiting_added,
                         distance_added, speed_added)
        env = CityFlowEnv(config)
        data_set_index = (data_set_index + 1) % len(args.scenarios_train)

        # Perform training run through environment.
        if smdp:
            run_env_smdp(agent, eps, config, env, "train")
        else:
            run_env_mdp(agent, eps, config, env, "train")

        # Decrease epsilon.
        decay = (EPS_START - EPS_END) / ((n_trajactories - starting_trajectory) * EPS_END_PERCENTAGE)
        eps = max(eps - decay, EPS_END)

        # Decrease learning rate.
        # agent.lr_scheduler.step()
        # lr = agent.lr_scheduler.get_last_lr()[0]
        lr = config['lr']

        # Save and show training stats.
        stats_every = 50
        if trajectory % stats_every == stats_every - 1:

            # Get training stats.
            for i, scenario in enumerate(args.scenarios_train):
                config_train_eval = setup_config(scenario, 'train', time, lr, batch_size, rm_size,
                                      learn_every, smdp, waiting_added,
                                      distance_added, speed_added)
                env_train_eval = CityFlowEnv(config_train_eval)
                stats_train = []

                # Perform evaluation run through environment on this training scenario.
                if smdp:
                    stats_one_training_run = run_env_smdp(agent, 0, config_train_eval, env_train_eval, "eval")
                    stats_train.append(stats_one_training_run)
                # else:
                #     stats_train = run_env_mdp(agent, 0, config, env, "eval")

                # Save logs of best run on this training scenario.
                if stats_one_training_run['travel_time'] < best_travel_time_train[i]:
                    print('BEST TRAIN on {}'.format(i))
                    env_train_eval.log()
                    best_travel_time_train[i] = stats_one_training_run['travel_time']

            average_travel_time_training_set = np.mean([x["travel_time"] for x in stats_train])
            average_reward_training_set = np.mean([x["rewards"] for x in stats_train])

            # Get validation stats.
            if smdp:
                stats_val = run_env_smdp(agent, 0, config_val, env_val, "eval")
            else:
                stats_val = run_env_mdp(agent, 0, config_val, env_val, "eval")

            # Save best model on validation set.
            if stats_val['travel_time'] < best_travel_time_val:
                print('BEST VAL\n')
                path = "{}/trained_models/{}/{}".format(args.output_dir, args.exp_name, config['hyperparams'])
                torch.save({
                    'stats': stats_train,
                    'model_state_dict': agent.qnetwork_local.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, os.path.join(path, "checkpoint.tar"))
                env_val.log()
                best_travel_time_val = stats_val['travel_time']

            # Evaluate the maximum Q values on a fixed set of random states.
            # average_max_qvals = eval_fixed_states(agent, 0, config)
            average_max_qvals = 0

            # Write important stats to Tensorboard.
            if TENSORBOARD:
                writer.add_scalar('Average Travel Time Train', average_travel_time_training_set, trajectory)
                writer.add_scalar('Average Travel Time VAL', stats_val['travel_time'], trajectory)
                writer.add_scalar('Eps', eps, trajectory)
                writer.add_scalar('LR', lr, trajectory)
                writer.add_scalar('Average Reward', average_reward_training_set / config['num_step'], trajectory)
                writer.add_scalar('Average Max Q Value', average_max_qvals, trajectory)
                for name, weight in agent.qnetwork_local.named_parameters():
                    writer.add_histogram(name + '_qnetwork_local', weight, trajectory)
                for name, weight in agent.qnetwork_target.named_parameters():
                    writer.add_histogram(name + '_qnetwork_target', weight, trajectory)

            # Print out stats in terminal.
            print(
                '\rTrajactory {}:\tTravel Time Train: {:.0f}\tTravel Time Val: {:.0f}\tMean Reward: {:.2f}\tBatch_size: {}\tLearning rate: {:.2g}\tRM '
                'size: {}\tLearn every: {}\tEpsilon: '
                '{:.2g}\tMax Q val: {:.1f}\tActions test: {}'.format(trajectory + 1,
                                                                                  average_travel_time_training_set,
                                                                                  stats_val['travel_time'],
                                                                                  average_reward_training_set / config[
                                                                                      'num_step'],
                                                                                  config['batch_size'], lr,
                                                                                  config["rm_size"],
                                                                                  config["learn_every"],
                                                                                  eps,
                                                                                  average_max_qvals,
                                                                                  stats_val['actions'].values()))

    # Load best model and evaluate on test set.
    config_test = setup_config(args.scenario_test, 'test', time, lr, batch_size, rm_size, learn_every, smdp, waiting_added,
                 distance_added, speed_added)
    env_test = CityFlowEnv(config_test)
    agent_test = Agent(state_size, action_size, seed=0)
    path = "{}/trained_models/{}/{}/checkpoint.tar".format(args.output_dir, args.exp_name, config['hyperparams'])
    checkpoint = torch.load(path)
    agent_test.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])

    stats_test = run_env_smdp(agent_test, 0, config_test, env_test, "eval")
    env_test.log()
    if TENSORBOARD:
        writer.add_scalar('Average Travel Time Test', stats_test['travel_time'], 0)

    # Create the replay logs.
    travel_times_training = []
    for i, scenario in enumerate(args.scenarios_train):
        config_train_eval = setup_config(scenario, 'train', time, lr, batch_size, rm_size,
                                         learn_every, smdp, waiting_added,
                                         distance_added, speed_added)
        travel_times_training.append(evaluate_one_traffic(config_train_eval))
    print("")
    print("====================== travel time ======================")
    print('train: average over multiple train sets: ' + ": {:.2f} s".format(np.mean(travel_times_training)))
    evaluate_one_traffic(config_val, 'print')
    evaluate_one_traffic(config_test, 'print')

    # Make sure that all pending events have been written to disk.
    if TENSORBOARD:
        writer.flush()
        writer.close()


def run_env_smdp(agent, eps, config, env, mode=None):
    """Run 1 episode through environment.

    Params
    ======
        agent (Agent): the DQN agent to train
        eps (float): value of epsilon for epsilon-greedy action selection
        config (json): configuration file to setup the CityFlow engine
        env (CityFlowEnv): CityFlow environment
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
             'travel_time': 0}

    t = 0
    state = env.reset()
    last_action, _ = agent.act(state, eps)

    while t < config['num_step']:
        action, _ = agent.act(state, eps)

        # Take step in environment, add yellow light if action changes.
        if action == last_action:
            next_state, reward = env.step(action)
        else:
            reward = 0
            counter = 0
            for _ in range(env.yellow_time):
                _, sub_reward = env.step(-1)  # action -1 -> yellow light
                reward += sub_reward * GAMMA ** counter
                stats['actions'][-1] += 1
                t += 1
                counter += 1

                # Break out of the training loop when training steps is reached.
                flag = (t >= config['num_step'])
                if flag:
                    break
            if flag:
                break
            next_state, sub_reward = env.step(action)
            reward += sub_reward * GAMMA ** counter

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


def run_env_mdp(agent, eps, config, env, mode=None):
    """Run 1 episode through environment.

    Params
    ======
        agent (Agent): the DQN agent to train
        eps (float): value of epsilon for epsilon-greedy action selection
        config (json): configuration file to setup the CityFlow engine
        env (CityFlowEnv): CityFlow environment
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
             'travel_time': 0}

    yellow_time = 0
    state = env.reset()
    last_action, _ = agent.act(state, eps)

    for t in range(config['num_step']):

        action, _ = agent.act(state, eps)

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

    run_env_smdp(random_agent, 1, config, env, 'random')
    env.log()
    evaluate_one_traffic(config, args.scenario, 'random', 'print')


def eval_fixed_states(agent, eps, config):
    if config['num_step'] == 300:
        random_states = [[1, 18, 1, 29, 0, 6, 2, 11, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0, 17, 1, 29, 0, 6, 1, 9, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         [0, 11, 1, 13, 0, 1, 1, 6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                         [1, 9, 2, 24, 0, 9, 0, 5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0, 9, 0, 17, 0, 2, 1, 5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    if config['num_step'] == 3600:
        if config['waiting_added'] == 0:
            random_states = [[2, 37, 1, 34, 2, 34, 0, 37, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0, 36, 0, 34, 2, 36, 0, 38, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0, 29, 3, 34, 2, 18, 0, 21, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0, 5, 2, 13, 0, 1, 0, 7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1, 12, 1, 27, 0, 6, 1, 11, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1, 7, 3, 10, 0, 0, 0, 5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0, 7, 3, 11, 0, 0, 0, 7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [1, 15, 2, 16, 0, 2, 1, 6, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1, 15, 4, 18, 0, 4, 0, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [1, 22, 6, 37, 0, 6, 1, 10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        if config['waiting_added'] == 1:
            random_states = [
                [1.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 13.0, 0.0, 18.0, 0.0, 7.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 4.0, 0.0, 2.0, 0.0, 0.0, 0.0, 21.0, 0.0, 26.0, 0.0, 2.0, 0.0, 9.0, 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 6.0, 1.0, 9.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 1.0, 3.0, 0.0, 2.0, 1.0, 2.0, 0.0, 12.0, 0.0, 29.0, 0.0, 4.0, 1.0, 11.0, 0.0, 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 1.0, 7.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 7.0, 0.0, 4.0, 0.0, 0.0, 0.0, 13.0, 0.0, 9.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.0, 0.0],
                [1.0, 8.0, 0.0, 9.0, 0.0, 0.0, 1.0, 9.0, 0.0, 5.0, 1.0, 13.0, 0.0, 6.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0]]

    qvals = 0
    for state in random_states:
        _, qval = agent.act(state, eps)
        qvals += qval
    average_max_qval = qvals / len(random_states)
    return average_max_qval
