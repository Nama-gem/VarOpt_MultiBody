#!/bin/bash

#SBATCH --job-name=varopt_multibody
#SBATCH --partition=jila
#SBATCH --qos=long

#SBATCH --array=0-14

#SBATCH --time=7-00:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

#SBATCH --output=./output/out-%x.%A_%a.out
#SBATCH --error=./output/err-%x.%A_%a.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@colorado.edu


# Load environment
module load anaconda/3.9
conda activate varopt39


# Optional: print some debugging information
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"


# Run optimization
srun python cluster_optimization.py