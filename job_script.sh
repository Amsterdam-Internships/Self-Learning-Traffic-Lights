#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 99:00:00
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
  --scenario_val "7.0.hangzou1_1x1_turns"\
  --scenario "7.1.hangzou2_1x1_turns"\
  --scenario_test "7.2.hangzou3_1x1_turns"\
  --exp_name 12.0.hangzou2_state_2.0\
  --num_step 3600\
  --trajectories 2000\
  --lrs "0.0001"\
  --batchsizes "528"\
  --output_dir "$TMPDIR"\
  --rm_size "360000"\
  --learn_every "4"\
  --smdp 0\
  --waiting_added "1"\

# copy checkpoints to home directory
mkdir -p $HOME/lisa_output
cp -r "$TMPDIR"/experiments $HOME/lisa_output
