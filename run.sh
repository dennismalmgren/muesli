#!/bin/bash
#SBATCH --reservation 1g.10gb
#SBATCH -t 2:00:00
#SBATCH -o runs/%j.out

# Parse command-line arguments
while getopts ":s:" opt; do
  case $opt in
    s)
      seed=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done


# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /proj/berzelius-aiics-real/users/x_denma/

# Execute my Apptainer image binding in the current working directory
# containing the Python script I want to execute

# for training baseline cmd="apptainer exec --env WANDB_API_KEY='f832ecbebaa081e6438201bd475fe26f9f0b1d82' --nv -B projs/muesli:/app berzdev_latest.sif bash -c 'cd /app/ppo && python ppo.py'"

# For generating random trajectories cmd="apptainer exec --env WANDB_API_KEY='f832ecbebaa081e6438201bd475fe26f9f0b1d82' --nv -B projs/muesli:/app berzdev_latest.sif bash -c 'cd /app/path_integration_module && python generate_random_trajectories.py'"

# For training predictors 
# For training predictors 
cmd="apptainer exec --env WANDB_API_KEY='f832ecbebaa081e6438201bd475fe26f9f0b1d82' --nv -B projs/muesli:/app berzdev_latest.sif bash -c 'cd /app && python -m cont_actions.rnad_cont_actions_blotto_expectation2'"

# for training solver 

# for training solver internal_cmd="energy_prediction.model_save_dir=$model logger.group_name=$model"
# for training solver cmd="apptainer exec --env WANDB_API_KEY='f832ecbebaa081e6438201bd475fe26f9f0b1d82' --nv -B projs/muesli:/app berzdev_latest.sif bash -c 'cd /app/ppo_grid && python ppo_grid.py $internal_cmd'"


eval $cmd
