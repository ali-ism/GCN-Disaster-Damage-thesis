#!/bin/bash

#SBATCH --job-name=dataset-cpu
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --time=1-00:00:00

module purge
module load python/pytorch-1.7.1

python3 dataset.py