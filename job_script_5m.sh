#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH -p gpu_shared

#Loading modules
#module load 2020
#module load Python

source activate sierk

# declare run
#experiment= "$TMPDIR"/test
echo "starting training run experiment"

mkdir "$TMPDIR"/experiments

# execute training script
cd Eigen/Eigen && pip install --user -e . && python $HOME/Eigen/Eigen/src/tuning.py\
  --scenario "7.0.hangzou1_1x1_turns"\
  --scenario_val "7.1.hangzou2_1x1_turns"\
  --scenario_test "7.2.hangzou3_1x1_turns"\
  --exp_name 12.0.hangzou1_state_2.0_5m\
  --num_step 300\
  --trajectories 1200\
  --lrs "0.01"\
  --batchsizes "128"\
  --output_dir "$TMPDIR"\
  --rm_size "20000"\
  --learn_every "4"\
  --smdp 1\
  --waiting_added "1"\
  --distance_added "1"\
  --speed_added "0"\

# copy checkpoints to home directory
mkdir -p $HOME/lisa_output
cp -r "$TMPDIR"/experiments $HOME/lisa_output
