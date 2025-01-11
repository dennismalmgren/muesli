#!/bin/bash
#SBATCH --reservation=1g.10gb
#SBATCH -t 02:00:00
#SBATCH -o runs/%j.out

# Apptainer images can only be used outside /home. In this example the
# image is located here
cd /proj/berzelius-aiics-real/users/x_denma/

apptainer exec --env "WANDB_API_KEY=f832ecbebaa081e6438201bd475fe26f9f0b1d82" --nv -B ./projs/muesli/muesli_work/tree_work/ensemble_prediction:/app berzdev_latest.sif bash -c "cd /app && python ppo_mujoco.py"
