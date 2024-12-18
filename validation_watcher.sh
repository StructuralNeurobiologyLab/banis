#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=500000
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

sleep 1m # waits for the checkpoint creation
mamba run -n nisb --no-capture-output python3 -u validation_watcher.py ${@}
