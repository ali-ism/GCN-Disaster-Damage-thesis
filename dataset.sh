#!/bin/bash

#SBATCH --job-name=dataset
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mem=8000
#SBATCH --time=0-06:00:00

module purge
module load cuda
#module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load python/pytorch-1.7.1

python3 dataset.py