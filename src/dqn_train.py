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

LOAD = 0  # Set to 1 to load checkpoint
RANDOM_RUN = 0   # (watch out, could overwrite replaylog of training)
TENSORBOARD = 1
EPS_START = 0.9
EPS_END = 0.1

args = parse_arguments()
# om hyperparam search op te zetten: https://deeplizard.com/learn/video/ycxulUVoNbk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()


def dqn(n_trajactories, config):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
    """

    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    # action_size = 2
    action_size = len(phase_list)
    state_size = len(CityFlowEnv(config).reset())
    best_travel_time = 100000

    log_dir = 'experiments/{}/tensorboard/{}'.format(args.exp_name, config['hyperparams']) + "_time=" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if TENSORBOARD:
        writer = SummaryWriter(log_dir, comment=f' batch_size={11} lr={0.1}')
        # save network structure
        writer.add_graph(Agent(state_size, action_size, 0).qnetwork_local,
                         torch.from_numpy(CityFlowEnv(config).reset()).unsqueeze(0).float())

    agent = Agent(state_size, action_size, 0, config['lr'])
    starting_trajectory = 0
    eps = EPS_START

    # Load saved checkpoint
    if LOAD == 1:
        checkpoint = torch.load("trained_models/{}/checkpoint.tar".format(args.exp_name))
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats = checkpoint['stats']
        # TODO load eps, maar werkt t wel om eps zo laag te laten bij trainen? misschien kan dit loaden en verder trainen niet echt.

    for trajectory in range(starting_trajectory + 1, n_trajactories + 1):
        # Perform training run through environment
        train_stats, _ = run_env(agent, eps, config, "train", trajectory)

        # Perform evaluation run through environment
        stats, env = run_env(agent, 0, config, "eval", trajectory)
        stats['loss'] = train_stats['loss']

        # Decrease epsilon
        decay = (EPS_START - EPS_END) / ((n_trajactories - starting_trajectory) * 0.8)
        eps = max(eps - decay, EPS_END)

        # Decrease learning rate
        agent.lr_scheduler.step()
        lr = agent.lr_scheduler.get_last_lr()[0]

        # Save best model
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

        print_every = 1
        if trajectory % print_every == print_every - 1:
            print('\rTrajactory {}\tMean Reward{:.2f}\tLoss {:.0f}\tLearning rate: {:.2g}\tEpsilon  {:.2g}\t Action count {}'
                  '\tTravel Time {:.0f}\tQ value size {:.0f}'.format(trajectory, stats['rewards']/(config['num_step'] - stats['actions'][-1]), stats['loss'], lr,
                                                eps,
                                                list(stats['actions'].values()),
                                                stats['travel_time'], np.mean(stats['q_values_size'])))

        if TENSORBOARD:
            writer.add_scalar('Eps', eps, trajectory)
            writer.add_scalar('LR', lr, trajectory)
            for name, weight in agent.qnetwork_local.named_parameters():
                writer.add_histogram(name + '_qnetwork_local', weight, trajectory)
            for name, weight in agent.qnetwork_target.named_parameters():
                writer.add_histogram(name + '_qnetwork_target', weight, trajectory)

            writer.add_scalar('Average Reward', stats['rewards']/(config['num_step'] - stats['actions'][-1]), trajectory)
            # todo add actions to 1 scalars plot, histgram doesnt show anything
            # writer.add_histogram('Actions', action, trajectory)
            writer.add_scalar('Loss', stats['loss'], trajectory)

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    # make sure that all pending events have been written to disk.
    if TENSORBOARD:
        writer.flush()
        writer.close()

    # to make the replay logs represent the best trajectory instead of the last trajectory.
    evaluate_one_traffic(config, args.scenario, 'train', 'print')


def run_env(agent, eps, config, mode=None, epoch=0):
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
            stats['loss'] += agent.loss

        # Save evaluation stats
        if mode == "eval":
            stats['actions'][action] += 1
            stats['rewards'] += reward

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

    test_state = np.array([5, 3, 5, 1, 8, 2, 3, 5, 0, 1, 0, 0, 0, 0, 0, 0])
    # test_state = np.array([10, 2, 3, 7, 0, 1, 0, 0, 0, 0, 0, 0])
    # test_state = np.array([10, 2, 3, 7, 1, 0])
    _, q_values = agent.act(test_state, 0)
    # todo return to other def
    # if TENSORBOARD:
    #     writer.add_scalar('Q value test state', q_values[0][0], epoch)
    #     writer.add_histogram('Q values test state', q_values[0], epoch)

    stats['travel_time'] = env.get_average_travel_time()
    return stats, env


# # Perform random run through environment before training
# if RANDOM_RUN == 1:
#     config = setup_config(NUM_STEPS, 'random')
#     _, environment = run_env(Agent(state_size, action_size, seed=0), 1)
#     environment.log()  # TODO why should this be before config change? how can config change the environment?
#     evaluate_one_traffic(config, args.scenario, 'random', 'print')
#     # print("reward and travel time when random, ", cum_reward, tt_random)
#     config = setup_config(NUM_STEPS, 'train', NORM_INPUTS, NORM_REWARDS, NORM_TAU)
#
# # Take average over training runs
# training_runs = []
# # als je deze gemiddeldes wil blijven bewaren moet je alles pas hierna naar de writer schrijven, en wat meer returnen.
# # of dit checken https://doneill612.github.io/Scalar-Summaries/
# for lr in LRS:
#     for i in range(TRAINING_RUNS):
#         training_runs.append(dqn(TRAJECTORIES, lr))
#
# # Evaluate last run and make ready for cleaner visualisation
# evaluate_one_traffic(config, args.scenario, 'train', 'print')


