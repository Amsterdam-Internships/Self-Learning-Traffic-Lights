#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 40:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sierkkanis@hotmail.com

#Loading modules
#module load 2020
#module load Python

# declare run
#experiment= "$TMPDIR"/test
echo "starting training run experiment"

mkdir "$TMPDIR"/experiments

# execute training script
cd Eigen/Eigen && python $HOME/Eigen/Eigen/src/tuning.py\
  --scenario "7.0.hangzou1_1x1_turns"\
  --exp_name 11.0.hangzou1_batch_size\
  --num_step 3600\
  --trajectories 2000\
  --lrs "0.001"\
  --batchsizes "528"\
  --output_dir "$TMPDIR"\
  --rm_size "360000"\
  --learn_every "8"

# copy checkpoints to home directory
mkdir -p $HOME/lisa_output
cp -r "$TMPDIR"/experiments $HOME/lisa_output
