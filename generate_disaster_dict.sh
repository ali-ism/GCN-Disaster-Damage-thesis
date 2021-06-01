#!/bin/bash

#SBATCH --job-name=disaster_dict
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=0-01:00:00

module purge
module load python/3.8.2

python3 generate_disaster_dict.py