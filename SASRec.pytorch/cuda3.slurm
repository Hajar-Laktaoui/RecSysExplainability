#!/bin/bash

#SBATCH --partition=gpu              # partition name
#SBATCH --job-name=single_gpu        # name of job 
#SBATCH --nodes=1                    # we request one node   
#SBATCH --ntasks-per-node=1          # with one task per node
#SBATCH --gres=gpu:1                 # number of GPUs (only one GPU per node is allowed for the `gpu` partition)
#SBATCH --cpus-per-task=1            # number of cores per task (>1 if multi-threaded tasks)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=48:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=gpu_single%j.out    # name of output file
#SBATCH --error=gpu_single%j.out     # name of error file (here, in common with the output file)

#nohup python UserEmbeddings.py --dataset=moviesAndTV --train_dir=default --maxlen=200 --device=cuda &
#python main.py --device=cuda --dataset=moviesAndTV --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=768.maxlen=200.pth' --inference_only=true --maxlen=200
python main.py --dataset=moviesAndTV --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda


sleep 48h
