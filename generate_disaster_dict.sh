#!/bin/bash

#SBATCH --job-name=disaster_dict
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --time=0-01:00:00

module purge
module load python/pytorch-1.7.1

python3 generate_disaster_dict.py