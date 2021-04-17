from itertools import product
import time

import torch
import cProfile, pstats, io

from src.dqn_train import *
from src.utility import *
from src.evaluate import *
from src.sotl_run import run_sotl
from src.cut_off import run_cut_off
from src.fixed_time import run_fixed_time

"""
This file contains the hyper-parameter loop.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)
print('version 2.0.1 MA test, works with single')  # To check if the right version is installed.
args = parse_arguments()

TRAIN = 1  # Boolean to train or not

TRAJECTORIES = args.trajectories

LRS = [float(item) for item in args.lrs.split(',')]
BATCH_SIZE = [int(item) for item in args.batchsizes.split(',')]
REPLAY_MEMORY_SIZE = [int(item) for item in args.rm_size.split(',')]
LEARN_EVERY = [int(item) for item in args.learn_every.split(',')]
WAITING_ADDED = [int(item) for item in args.waiting_added.split(',')]
DISTANCE_ADDED = [int(item) for item in args.distance_added.split(',')]
SPEED_ADDED = [int(item) for item in args.speed_added.split(',')]

# Make a list of hyper params to tune
hyper_parameters = dict(lrs=LRS, batch_size=BATCH_SIZE, rm_size=REPLAY_MEMORY_SIZE, learn_every=LEARN_EVERY,
                        waiting_added=WAITING_ADDED, distance_added=DISTANCE_ADDED, speed_added=SPEED_ADDED)
param_values = [v for v in hyper_parameters.values()]
# 0.0001 32 kan ook, die is in principe de langzaamste leerder, maar leert misschien wel het langst door.
# 0.001/0.0001 met 128 best. 0.0001 ook met 32 oke, misschien kleine lr beter met grotere replay?
# ook eens nog wat kleinere lr proberen.
# met grote replay werkt 0.01 iig niet meer.

def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

# @profile
def main():
    # Train the Deep Reinforcement Learning agent with the list of hyper parameters provided.
    if TRAIN:
        for lr, batch_size, rm_size, learn_every, waiting_added, distance_added, speed_added in product(*param_values):
            normalized_trajectories = TRAJECTORIES * learn_every
            start = time.time()
            time_clean = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            dqn(normalized_trajectories, time_clean, lr, batch_size, rm_size, learn_every, args.smdp, waiting_added, distance_added, speed_added)
            end = time.time()
            print("\nThis training loop took this amount of seconds: ", end-start)

    # TODO this should be part of another 'compare' file, in which perhaps the agent is reloaded with the saved network.
    # Compare the Deep Reinforcement Learning agent with baseline methods.

    # travel_times_training_set_sotl = []
    # for i, scenario in enumerate(args.scenarios_train):
    #     config_train = setup_config(scenario, 'sotl_train')
    #     travel_times_training_set_sotl.append(run_sotl(config_train, scenario))
    # print("")
    # print("====================== travel time ======================")
    # print('sotl_train: average over multiple train sets: ' + ": {:.2f} s".format(np.mean(travel_times_training_set_sotl)))
    #
    # config_val = setup_config(args.scenario_val, 'sotl_val')
    # run_sotl(config_val, args.scenario_val)
    #
    # config_test = setup_config(args.scenario_test, 'sotl_test')
    # run_sotl(config_test, args.scenarios_train[8])

    # config_fixed = setup_config(args.scenario_test, 'fixed_time', time=0, lr=0, batch_size=0, rm_size=0, learn_every=0, smdp=0, waiting_added=0,
    #              distance_added=0, speed_added=0)
    # run_fixed_time(config_fixed)

    # config_random = setup_config(args.scenario_test, 'random', distance_added=args.distance_added,
    #                              waiting_added=args.waiting_added, smdp=args.smdp)
    # random_run(config_random)

    # Show travel time per simulation second.
    # travel_time_plot()


if __name__ == "__main__":
    main()
