from src.dqn_train import dqn
from src.utility import *
from src.sotl_run import run_sotl

"""
This file contains the hyper parameter loops.
"""

TRAJECTORIES = 1
# LRS = [1e-2, 1e-3, 1e-4, 1e-5]
LRS = [1e-3]
TRAIN = 1

args = parse_arguments()


def main():
    if TRAIN:
        for lr in LRS:
            config = setup_config('train', 'train', lr, 0, 0)
            dqn(TRAJECTORIES, config)
    run_sotl()


if __name__ == "__main__":
    main()