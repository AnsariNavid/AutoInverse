#!/bin/bash
#SBATCH -J na_train_forward_model
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=0-0
#SBATCH -o /HPS/uana/work/slurm_log/%x-%j-%a.log
#SBATCH -e /HPS/uana/work/slurm_log/%x-%j-%a.err

configFile=""
method="na"
step="train_forward_model"

python main.py --config $configFile --method $method --step $step
