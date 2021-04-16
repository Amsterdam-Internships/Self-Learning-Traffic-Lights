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
LOAD = 0  # Set to 1 to load checkpoint
EPS_START = 1
EPS_END = 0.1
EPS_END_PERCENTAGE = 0.1
GAMMA = 0.99  # discount factor (Should be same as in agent file)
STATS_EVERY = 50

args = parse_arguments()


def dqn(n_trajactories, time, lr, batch_size, rm_size, learn_every, smdp, waiting_added, distance_added, speed_added):
    """ Deep Q-Learning

    Params
    ======
        n_trajactories (int): maximum number of training episodes
        config (json): configuration file to setup the CityFlow engine
    """

    config_train = []
    envs_train = []
    for scenario in args.scenarios_train:
        config = setup_config(scenario, 'train', time, lr, batch_size, rm_size,
                                         learn_every, smdp, waiting_added,
                                         distance_added, speed_added)
        config_train.append(config)
        envs_train.append(CityFlowEnv(config))

    config_val = setup_config(args.scenario_val, 'val', time, lr, batch_size, rm_size, learn_every, smdp, waiting_added,
                 distance_added, speed_added)
    env_val = CityFlowEnv(config_val)

    # intersection_id = list(config_val['lane_phase_info'].keys())[0]
    # phase_list = config_val['lane_phase_info'][intersection_id]['phase']
    best_travel_time_train = np.ones(len(args.scenarios_train)) * 1000000
    best_travel_time_val = 100000
    starting_trajectory = 0
    eps = EPS_START
    data_set_index = 0

    agents = [None for i in config['intersection_indices']]
    for i, intersection_index in enumerate(config['intersection_indices']):
        if config['acyclic']:
            action_size = len(env_val.phase_lists[intersection_index])
        else:
            action_size = 2
        state_size = len(env_val.get_state(intersection_index))
        agents[i] = Agent(state_size, action_size, 0, lr, batch_size, rm_size,
                  learn_every)

    if TENSORBOARD:
        log_dir = '{}/experiments/{}/tensorboard/{}'.format(args.output_dir, args.exp_name, config_val[
            'hyperparams']) + "_time=" + time
        writer = SummaryWriter(log_dir, comment=f' batch_size={11} lr={0.1}')

    for trajectory in range(starting_trajectory + 1, n_trajactories + 1):
        config = config_train[data_set_index]
        env = envs_train[data_set_index]
        data_set_index = (data_set_index + 1) % len(args.scenarios_train)

        # Perform training run through environment.
        if smdp:
            run_env_smdp(agents, eps, config, env, "train")
        else:
            run_env_mdp(agents, eps, config, env, "train")

        # Decrease epsilon.
        decay = (EPS_START - EPS_END) / ((n_trajactories - starting_trajectory) * EPS_END_PERCENTAGE)
        eps = max(eps - decay, EPS_END)

        # Save and show training stats.
        if trajectory % STATS_EVERY == STATS_EVERY - 1:

            # Get training stats.
            for i in range(len(config_train)):

                stats_train = []
                # Perform evaluation run through environment on this training scenario.
                if smdp:
                    stats_one_training_run = run_env_smdp(agents, 0, config_train[i], envs_train[i], "eval")
                    stats_train.append(stats_one_training_run)
                else:
                    stats_one_training_run = run_env_mdp(agents, 0, config_train[i], envs_train[i], "eval")
                    stats_train.append(stats_one_training_run)

                # Save logs of best run on this training scenario.
                if stats_one_training_run['travel_time'] < best_travel_time_train[i]:
                    print('BEST TRAIN on {}, {} s'.format(i, stats_one_training_run['travel_time']))
                    envs_train[i].log()
                    best_travel_time_train[i] = stats_one_training_run['travel_time']

            average_travel_time_training_set = np.mean([x["travel_time"] for x in stats_train])
            average_reward_training_set = np.mean([x["rewards"] for x in stats_train])

            # Get validation stats.
            if smdp:
                stats_val = run_env_smdp(agents, 0, config_val, env_val, "eval")
            else:
                stats_val = run_env_mdp(agents, 0, config_val, env_val, "eval")

            # Save best model on validation set.
            if stats_val['travel_time'] < best_travel_time_val:
                print('BEST VAL\n')
                path = "{}/trained_models/{}/{}".format(args.output_dir, args.exp_name, config['hyperparams'])
                for i, agent in enumerate(agents):
                    path = path + "/checkpoint" + str(i) + ".tar"
                    torch.save({
                        # 'stats': stats_train,
                        'model_state_dict': agent.qnetwork_local.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, path)
                env_val.log()
                best_travel_time_val = stats_val['travel_time']

            # Evaluate the maximum Q values on a fixed set of random states.
            if config['smdp'] == 1 and config['speed_added'] == 0 and config['acyclic'] == 1 and config['waiting_added'] == 1\
                    and config['distance_added'] == 1 and config['num_step'] == 3600 and config['multi_agent'] == 0:
                average_max_qvals = eval_fixed_states(agents[0], 0, config)
            else:
                average_max_qvals = 0

            # Write important stats to Tensorboard.
            if TENSORBOARD:
                writer.add_scalar('Average Travel Time Train', average_travel_time_training_set, trajectory)
                writer.add_scalar('Average Travel Time VAL', stats_val['travel_time'], trajectory)
                writer.add_scalar('Eps', eps, trajectory)
                writer.add_scalar('LR', lr, trajectory)
                writer.add_scalar('Average Reward', average_reward_training_set / config['num_step'], trajectory)
                writer.add_scalar('Average Max Q Value', average_max_qvals, trajectory)
                for name, weight in agents[0].qnetwork_local.named_parameters():
                    writer.add_histogram(name + '_qnetwork_local', weight, trajectory)
                for name, weight in agents[0].qnetwork_target.named_parameters():
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

    agents_test = [None for i in config['intersection_indices']]
    for i, intersection_index in enumerate(config['intersection_indices']):
        if config['acyclic']:
            action_size = len(env_val.phase_lists[intersection_index])
        else:
            action_size = 2
        state_size = len(env_val.get_state(intersection_index))
        agents_test[i] = Agent(state_size, action_size, seed=0)
        path = "{}/trained_models/{}/{}/checkpoint{}.tar".format(args.output_dir, args.exp_name, config['hyperparams'], i)
        checkpoint = torch.load(path)
        agents_test[i].qnetwork_local.load_state_dict(checkpoint['model_state_dict'])

    if smdp:
        stats_test = run_env_smdp(agents_test, 0, config_test, env_test, "eval")
    else:
        stats_test = run_env_mdp(agents_test, 0, config_test, env_test, "eval")

    env_test.log()
    if TENSORBOARD:
        writer.add_scalar('Average Travel Time Test', stats_test['travel_time'], 0)

    # Create the replay logs.
    travel_times_training = []
    for i in range(len(config_train)):
        travel_times_training.append(evaluate_one_traffic(config_train[i]))
    print("")
    print("====================== travel time ======================")
    print('train: average over multiple train sets: ' + ": {:.2f} s".format(np.mean(travel_times_training)))
    evaluate_one_traffic(config_val, 'print')
    evaluate_one_traffic(config_test, 'print')

    # Make sure that all pending events have been written to disk.
    if TENSORBOARD:
        writer.flush()
        writer.close()


def run_env_smdp2(agents, eps, config, env, mode=None):
    """Run 1 episode through environment.

    Params
    ======
        agents (Agent): the DQN agents to train
        eps (float): value of epsilon for epsilon-greedy action selection
        config (json): configuration file to setup the CityFlow engine
        env (CityFlowEnv): CityFlow environment
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
             'travel_time': 0}
    env.reset()

    agent = agents[0]
    intersection_index = 0

    # Initialize the last_actions array, to check if new action is same or different, and add yellow light accordingly.
    state = env.get_state(env.intersection_indices[intersection_index])
    if config['acyclic']:
        last_action, _ = agent.act(state, eps)
    else:
        last_action = 0

    action = 0
    yellow_light_reward = 0
    yellow_light_counter = 0
    for t in range(config['num_step']):

        # Choose an action at every intersection.
        # state = env.get_state(env.intersection_indices[0])

        # If light is not on yellow, make an action and check if light switches or not.
        # If it switches, perform that action only after yellow time is finished.
        if yellow_light_counter == 0:
            action, _ = agent.act(state, eps)
            if action == last_action:
                env.step(action, intersection_index)
            else:
                env.step(-1, intersection_index)
        elif 0 < yellow_light_counter < env.yellow_time:
            env.step(-1, intersection_index)
        elif yellow_light_counter == env.yellow_time:
            env.step(action, intersection_index)

        # If light is not on yellow, check if light switches.
        # If it switches, record all the subrewards it gets during yellow period.
        if yellow_light_counter == 0:
            if action == last_action:
                reward = env.get_reward(intersection_index)
            else:
                sub_reward = env.get_reward(intersection_index)
                yellow_light_reward = sub_reward
                yellow_light_counter += 1
                # continue  # Makes sure the agent is not training during yellow time.
        elif 0 < yellow_light_counter < env.yellow_time:
            sub_reward = env.get_reward(intersection_index)
            yellow_light_reward += sub_reward * GAMMA ** yellow_light_counter
            yellow_light_counter += 1
            # continue  # Makes sure the agent is not training during yellow time.
        elif yellow_light_counter == env.yellow_time:
            sub_reward = env.get_reward(intersection_index)
            reward = yellow_light_reward + sub_reward * GAMMA ** yellow_light_counter
            yellow_light_counter = 0

        if yellow_light_counter == 0:
            next_state = env.get_state(env.intersection_indices[0])
            # Add to replay buffer and train.
            if mode == "train":
                agent.step(state, action, reward, next_state)

            # Save evaluation stats.
            if mode == "eval":
                stats['rewards'] += reward

            state = next_state
            if config['acyclic']:
                last_action = action
            else:
                last_action = 0

    stats['travel_time'] = env.get_average_travel_time()
    return stats


def run_env_smdp(agents, eps, config, env, mode=None):
    """Run 1 episode through environment.

    Params
    ======
        agents (Agent): the DQN agents to train
        eps (float): value of epsilon for epsilon-greedy action selection
        config (json): configuration file to setup the CityFlow engine
        env (CityFlowEnv): CityFlow environment
        mode (string): agent only takes step on 'train' mode
    """
    stats = {'rewards': 0, 'actions': {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
             'travel_time': 0}
    env.reset()

    # Initialize the last_actions array, to check if new action is same or different, and add yellow light accordingly.
    last_actions = [0 for i in agents]
    states = [[] for i in agents]
    for i, agent in enumerate(agents):
        states[i] = env.get_state(env.intersection_indices[i])
        if config['acyclic']:
            last_actions[i], _ = agent.act(states[i], eps)
        else:
            last_actions[i] = 0

    actions = [0 for i in agents]
    yellow_light_rewards = [0 for i in agents]
    yellow_light_counters = [0 for i in agents]
    for t in range(config['num_step']):

        # Choose an action at every intersection.
        for i, intersection_index in enumerate(env.intersection_indices):

            # If light is not on yellow, make an action and check if light switches or not.
            # If it switches, perform that action only after yellow time is finished.
            if yellow_light_counters[i] == 0:
                actions[i], _ = agents[i].act(states[i], eps)
                if actions[i] == last_actions[i]:
                    env.step(actions[i], intersection_index)
                else:
                    env.step(-1, intersection_index)
            elif 0 < yellow_light_counters[i] < env.yellow_time:
                env.step(-1, intersection_index)
            elif yellow_light_counters[i] == env.yellow_time:
                env.step(actions[i], intersection_index)

        # When all agents have chosen their action, record their rewards and next states.
        for i, intersection_index in enumerate(env.intersection_indices):

            # If light is not on yellow, check if light switches.
            # If it switches, record all the subrewards it gets during yellow period.
            if yellow_light_counters[i] == 0:
                if actions[i] == last_actions[i]:
                    reward = env.get_reward(intersection_index)
                else:
                    sub_reward = env.get_reward(intersection_index)
                    yellow_light_rewards[i] = sub_reward
                    yellow_light_counters[i] += 1
            elif 0 < yellow_light_counters[i] < env.yellow_time:
                sub_reward = env.get_reward(intersection_index)
                yellow_light_rewards[i] += sub_reward * GAMMA ** yellow_light_counters[i]
                yellow_light_counters[i] += 1
            elif yellow_light_counters[i] == env.yellow_time:
                sub_reward = env.get_reward(intersection_index)
                reward = yellow_light_rewards[i] + sub_reward * GAMMA ** yellow_light_counters[i]
                yellow_light_counters[i] = 0

            if yellow_light_counters[i] == 0:
                next_state = env.get_state(env.intersection_indices[i])
                # Add to replay buffer and train.
                if mode == "train":
                    agents[i].step(states[i], actions[i], reward, next_state)

                # Save evaluation stats.
                if mode == "eval":
                    stats['rewards'] += reward

                if config['acyclic']:
                    last_actions[i] = actions[i]
                else:
                    last_actions[i] = 0

                states[i] = next_state

    stats['travel_time'] = env.get_average_travel_time()
    return stats


def run_env_smdp3(agents, eps, config, env, mode=None):
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

    agent = agents[0]

    t = 0
    state = env.reset()

    # lijsten van 4
    if config['acyclic']:
        last_action, _ = agent.act(state, eps)
    else:
        last_action = 0

    while t < config['num_step']:
        # for lloop
        #lijst van 4
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
        if config['acyclic']:
            last_action = action
        else:
            last_action = 0
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


def random_run(config):
    """ Perform random run through environment without training
    """
    env = CityFlowEnv(config)

    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    action_size = len(phase_list)
    state_size = len(env.reset())
    random_agent = Agent(state_size, action_size, seed=0)

    run_env_smdp(random_agent, 1, config, env, 'random')
    env.log()
    evaluate_one_traffic(config, 'print')


def eval_fixed_states(agent, eps, config):
    if config['num_step'] == 3600:
        if config['waiting_added'] == 0:
            random_states = []
        if config['waiting_added'] == 1 and config['distance_added'] == 0:
            random_states = []
        if config['waiting_added'] == 1 and config['distance_added'] == 1:
            random_states = [
                [0.025, -0.29999999999999993, 0.025, 0.016666666666666607, 0.0, -0.09166666666666667,
                 0.01666666666666667, -0.20833333333333326, 0.0, 1.2, 0.0, 0.8333333333333334, 0.0,
                 0.9666666666666667, 0.03333333333333333, 1.1333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.7851166666666667, 0.4765228899074073, 0.2296166666666667,
                 0.48519527450980393, 0.0, 0.49379644847619064, 0.5968638883333334, 0.49578811180180177],
                [0.025, -0.23333333333333328, 0.025, -0.09166666666666667, 0.06666666666666668,
                 -0.15833333333333344, 0.0, -0.1333333333333333, 0.0, 1.1333333333333333, 0.0, 0.9666666666666667,
                 0.03333333333333333, 1.0333333333333334, 0.0, 0.7333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.11851666666666666, 0.5031534383333334, 0.6369833333333333,
                 0.4899376138095239, 0.8499849158333334, 0.46151153123809524, 0.0, 0.6552679136111111],
                [-0.008333333333333331, -0.30833333333333335, 0.01666666666666667, 0.016666666666666607, 0.0,
                 -0.008333333333333304, 0.0, 0.04166666666666663, 0.03333333333333333, 1.2333333333333334,
                 0.03333333333333333, 0.8333333333333334, 0.1, 0.5333333333333333, 0.0, 0.43333333333333335, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.96411111, 0.49191507540540536,
                 0.5968638883333334, 0.49456475245098036, 0.9265698158333333, 0.6463785977777777, 0.0,
                 0.7032629038596492],
                [-0.033333333333333326, -0.15000000000000002, 0.0, -0.13333333333333341, 0.0, -0.20000000000000007,
                 -0.008333333333333331, -0.025000000000000022, 0.13333333333333333, 1.0, 0.0, 1.0333333333333334,
                 0.0, 1.1, 0.03333333333333333, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.9242905958333334, 0.48580932068627447, 0.0, 0.4731724755555555, 0.0, 0.4849270236111112,
                 0.9639494999999999, 0.4945507948571428],
                [-0.008333333333333331, -0.016666666666666607, 0.0, 0.04999999999999999, 0.0, 0.04999999999999999,
                 0.0, 0.09166666666666667, 0.03333333333333333, 0.5666666666666667, 0.0, 0.1, 0.0, 0.1, 0.0,
                 0.03333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.96411111,
                 0.6251536860606061, 0.0, 0.8898174988888891, 0.0, 0.8896807855555556, 0.0, 0.8035442666666668],
                [0.0, 0.1416666666666667, 0.025, 0.125, 0.0, 0.1, 0.025, 0.025, 0.0, 0.13333333333333333, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.5633646663636364, 0.03, 0.6180455553333334, 0.0, 0.48885, 0.5629166666666666, 0.5629166666666666],
                [0.0, 0.06666666666666676, -0.016666666666666663, 0.016666666666666607, 0.058333333333333334,
                 -0.20833333333333326, 0.01666666666666667, -0.14166666666666672, 0.0, 0.7333333333333333,
                 0.06666666666666667, 0.8333333333333334, 0.06666666666666667, 1.1333333333333333,
                 0.03333333333333333, 0.7666666666666667, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.4655956048958334, 0.9516111100000001, 0.5059834267647059, 0.5720648413333335,
                 0.4791930006306306, 0.8190638883333334, 0.623729152],
                [0.025, -0.20000000000000007, 0.05, 0.125, 0.041666666666666664, 0.07499999999999996, 0.0,
                 -0.15833333333333333, 0.0, 1.1, 0.0, 0.7, 0.03333333333333333, 0.5, 0.0, 0.6333333333333333, 0.0,
                 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8591833333333333, 0.4945985242592594,
                 0.45246108333333335, 0.4960863610101009, 0.7994148144444444, 0.609645196231884, 0.0,
                 0.7320565757894737],
                [0.05, 0.09166666666666667, 0.008333333333333331, -0.06666666666666665, 0.0, -0.016666666666666663,
                 0.0, -0.033333333333333326, 0.0, 0.7333333333333333, 0.06666666666666667, 0.6666666666666666, 0.0,
                 0.26666666666666666, 0.0, 0.23333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                 0.0, 0.0, 0.6555, 0.5090833513131313, 0.8224421299999999, 0.6174349819444445, 0.0,
                 0.7576872659999999, 0.0, 0.8750049016666666],
                [0.025, 0.19999999999999998, -0.02500000000000001, -0.06666666666666665, 0.0, -0.033333333333333326,
                 0.0, 0.075, 0.0, 0.1, 0.1, 0.36666666666666664, 0.0, 0.23333333333333334, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.26665, 0.6575225941666667, 0.93911111,
                 0.8191810175000002, 0.0, 0.85192361, 0.0, 0.07073888888888889]
            ]

    qvals = 0
    for state in random_states:
        _, qval = agent.act(state, eps)
        qvals += qval
    average_max_qval = qvals / len(random_states)
    return average_max_qval
