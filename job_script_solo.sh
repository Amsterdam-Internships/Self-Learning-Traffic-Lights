#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=END
#SBATCH --mail-user=sierkkanis@hotmail.com

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
  --scenarios_train "hangzhou_1x1_1h_A3"\
  --scenario_val "hangzhou_1x1_1h_A2"\
  --scenario_test "hangzhou_1x1_1h_A1"\
  --exp_name 13.0.hangzou3_state_2.0_multi_train\
  --num_step 3600\
  --trajectories 2000\
  --lrs "0.001"\
  --batchsizes "528"\
  --output_dir "$TMPDIR"\
  --rm_size "360000"\
  --learn_every "4"\
  --smdp 1\
  --waiting_added "0"\
  --distance_added "0"\
  --speed_added "0"\
  --acyclic "1"\

# copy checkpoints to home directory
mkdir -p $HOME/lisa_output
cp -r "$TMPDIR"/experiments $HOME/lisa_output
cp -r "$TMPDIR"/trained_models $HOME/lisa_output
