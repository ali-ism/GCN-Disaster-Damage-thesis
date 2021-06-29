#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=0-01:00:00

source ~/.bashrc

module purge
module load python/3.8.2

JUPYTER_PORT=$(random_unused_port)

jupyter-lab  --no-browser --port=${JUPYTER_PORT} > jupyter-${SLURM_JOB_ID}.log 2>&1 &
ssh -R localhost:${JUPYTER_PORT}:localhost:${JUPYTER_PORT} ohead1 -N