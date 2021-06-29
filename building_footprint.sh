#!/bin/bash

#SBATCH --job-name=building_footprint
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=0-02:00:00

module purge
module load python/3.8.2
module load torch/1.7.1-py38-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load cuda

python3 -u building_footprint.py