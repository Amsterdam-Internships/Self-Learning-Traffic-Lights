from itertools import product

import torch

from src.dqn_train import dqn
from src.utility import *
from src.sotl_run import run_sotl

"""
This file contains the hyper parameter loops.
"""

TRAJECTORIES = 3000

# LRS = [1e-3]
LRS = [1e-2, 1e-3, 1e-4, 1e-5]
BATCH_SIZE = [16, 32, 64, 128]
TRAIN = 1

hyper_parameters = dict(
    lrs=LRS,
    batch_size=BATCH_SIZE
)
param_values = [v for v in hyper_parameters.values()]

# args = parse_arguments()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)


def main():
    if TRAIN:
        for lr, batch_size in product(*param_values):
            config = setup_config('train', 'train', lr, batch_size, norm_inputs=0, norm_rewards=0)
            dqn(TRAJECTORIES, config)
    run_sotl()


if __name__ == "__main__":
    main()
