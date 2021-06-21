#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=arza

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --time=0-01:00:00

source ~/.bashrc

module purge
module load python/3.8.2
module load torch/1.7.1-py38-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load cuda

JUPYTER_PORT=$(random_unused_port)

jupyter-lab  --no-browser --port=${JUPYTER_PORT} > jupyter-${SLURM_JOB_ID}.log 2>&1 &
ssh -R localhost:${JUPYTER_PORT}:localhost:${JUPYTER_PORT} ohead1 -N