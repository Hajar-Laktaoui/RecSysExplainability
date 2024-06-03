#!/bin/bash

#SBATCH --partition=gpu              # partition name
#SBATCH --job-name=single_gpu        # name of job 
#SBATCH --nodes=1                    # we request one node   
#SBATCH --ntasks-per-node=1          # with one task per node
#SBATCH --gres=gpu:1                 # number of GPUs (only one GPU per node is allowed for the `gpu` partition)
#SBATCH --cpus-per-task=1            # number of cores per task (>1 if multi-threaded tasks)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=48:00:00 
#SBATCH --output=gpu_single%j.out    # name of output file
#SBATCH --error=gpu_single%j.out     # name of error file (here, in common with the output file)
#SBATCH -x node13


# source ~/.bashrc
module load cuda11/toolkit/11.7.0

python -u main.py --data_path ../MoviesAndTV/reviews.pickle --index_dir ../MoviesAndTV/1/ --cuda --checkpoint ./moviesAndTV/ >> moviesAndTV.log

sleep 48h
