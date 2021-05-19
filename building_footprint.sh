#!/bin/bash

#SBATCH --job-name=building_footprint
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=12000
#SBATCH --time=0-01:00:00

module purge
module load python/ai-2

python3 building_footprint.py