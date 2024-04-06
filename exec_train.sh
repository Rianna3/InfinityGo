#!/bin/bash

# SLURM config
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --gpu-bind=map_gpu:7
#SBATCH --mem=28g

#SBATCH -p "ug-gpu-small"
#SBATCH --qos="long-low-prio"
#SBATCH -t 01-00:00:00

#SBATCH --job-name=GO
#SBATCH -o GO.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user flnf58@durham.ac.uk

# venv config
conda activate GO

# execute
python main.py

