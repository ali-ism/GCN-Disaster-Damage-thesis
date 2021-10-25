#!/bin/bash

#SBATCH --job-name=ttest
#SBATCH --partition=msfea-ai

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=32G

module purge
module load python/tensorflow-2.3.1
module load cuda

python3 -u ttest_ae.py