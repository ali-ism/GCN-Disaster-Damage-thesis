#!/bin/bash

#SBATCH --job-name=building_footprint
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=0-02:00:00

module purge
module load python/pytorch-1.7.1

python3 building_footprint.py