from itertools import product
import time

import torch
import cProfile, pstats, io

from src.dqn_train import *
from src.utility import *
from src.evaluate import *
from src.sotl_run import run_sotl
# from src.sotl_LIT import run_sotl_LIT

"""
This file contains the hyper-parameter loop.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)
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


def main():

    # Train the Deep Reinforcement Learning agent with the list of hyper parameters provided.
    if TRAIN:
        print('version 1.0.0')  # To check if the right version is installed.
        for lr, batch_size, rm_size, learn_every, waiting_added, distance_added, speed_added in product(*param_values):
            config = setup_config('train', 'train', lr, batch_size, rm_size, learn_every, args.smdp, waiting_added, distance_added, speed_added)
            config_val = setup_config('val', 'val', lr, batch_size, rm_size, learn_every, args.smdp, waiting_added, distance_added, speed_added)
            config_test = setup_config('test', 'test', lr, batch_size, rm_size, learn_every, args.smdp, waiting_added, distance_added, speed_added)
            normalized_trajectories = TRAJECTORIES * learn_every

            start = time.time()
            dqn(normalized_trajectories, config, config_val, config_test)
            end = time.time()
            print("\nThis training loop took this amount of seconds: ", end-start)

    # Compare the Deep Reinforcement Learning agent with baseline methods.
    config_train = setup_config('train', 'sotl_train')
    run_sotl(config_train)
    config_val = setup_config('val', 'sotl_val')
    run_sotl(config_val)
    config_test = setup_config('test', 'sotl_test')
    run_sotl(config_test)
    # run_sotl_LIT()
    # random_run()

    # Show travel time per simulation second.
    # travel_time_plot()


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

if __name__ == "__main__":
    main()
