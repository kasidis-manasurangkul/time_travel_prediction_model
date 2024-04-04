#!/bin/bash 
 
#SBATCH --job-name=time_travel_prediction_model
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j_%N.log 

#SBATCH --mem=128gb  
#SBATCH --cpus-per-task=32 
#SBATCH --tasks-per-node=1

module purge
#module load miniconda3-4.10.3-gcc-9.3.0-u6p3tgr
/home/kmanasu/.conda/envs/time_travel_prediction_model/bin/python model.py
