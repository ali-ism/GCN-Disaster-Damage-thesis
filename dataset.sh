#!/bin/bash

#SBATCH --job-name=train
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=12000
#SBATCH --time=0-01:00:00

module purge
module load cuda

if [ nvidia-smi | sed '8,4!d' | awk '{print $4}' = "k20n" ]
then
module load torch/1.7.1-k20-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
else
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
fi

module load python/ai-2

python3 dataset.py