from itertools import product

import torch

from src.dqn_train import *
from src.utility import *
from src.evaluate import *
from src.sotl_run import run_sotl
from src.sotl_LIT import run_sotl_LIT

"""
This file contains the hyper-parameter loops.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)
args = parse_arguments()

TRAIN = 1  # Boolean to train or not

TRAJECTORIES = 4000
LRS = [1e-2]
BATCH_SIZE = [32, 128]

# Make a list of hyper params to tune
hyper_parameters = dict(lrs=LRS, batch_size=BATCH_SIZE)
param_values = [v for v in hyper_parameters.values()]
# product = [[0.01, 32], [0.01, 64], [0.01, 128], [0.001, 16], [0.001, 32], [0.001, 64], [0.001, 128], [0.0001, 16], [0.0001, 32], [0.0001, 64]]
# product = [[0.01, 128], [0.03, 128], [0.001, 32], [0.001, 64], [0.0001, 32], [0.0001, 64]]
# product = [[0.01, 64], [0.01, 128], [0.001, 32], [0.001, 64], [0.001, 128], [0.0001, 32], [0.0001, 64]]
# product = [[0.001, 128]]
# 0.0001 32 kan ook, die is in principe de langzaamste leerder, maar leert misschien wel het langst door.

def main():

    # Train the Deep Reinforcement Learning agent with the list of hyper parameters provided.
    if TRAIN:
        for lr, batch_size in product(*param_values):
            config = setup_config('train', 'train', lr, batch_size, norm_inputs=0, norm_rewards=0)
            dqn(TRAJECTORIES, config)

    # Compare the Deep Reinforcement Learning agent with baseline methods.
    run_sotl()
    # run_sotl_LIT()
    # random_run()

    # config = setup_config('train', 'train', 0.01, 128, norm_inputs=0, norm_rewards=0)
    # evaluate_one_traffic(config, args.scenario, 'train', 'print')

    # Show travel time per simulation second.
    # travel_time_plot()


if __name__ == "__main__":
    main()
