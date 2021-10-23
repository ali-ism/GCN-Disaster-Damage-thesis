#!/bin/bash

#SBATCH --job-name=ttest
#SBATCH --partition=msfea-ai

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=32G

module purge
module load python/3.8.2
module load torch/1.7.1-py38-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load cuda

python3 ttest_gcn.py

module purge
module load python/tensorflow-2.3.1
module load cuda

python3 ttest_ae.py

python3 ttest.py