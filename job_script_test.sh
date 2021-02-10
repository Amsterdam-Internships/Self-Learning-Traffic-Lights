#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -p gpu_short

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
  --exp_name tesy4\
  --num_step 3600\
  --trajectories 3\
  --lrs "0.001"\
  --batchsizes "512"\
  --output_dir "$TMPDIR"\
  --rm_size "36000,360000"\
  --learn_every "4"

# copy checkpoints to home directory
mkdir -p $HOME/lisa_output
cp -r "$TMPDIR"/experiments $HOME/lisa_output
