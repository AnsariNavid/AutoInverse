#!/bin/bash

numberOfModels=10
configFile=""
method="train_mu_model"
#SBATCH -J (configFile)_train_mu_model
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=1-1
#SBATCH -o slurm_log/%x_%j_%a.log
#SBATCH -e slurm_log/%x_%j_%a.err

for (( model_id = 0; model_id < $numberOfModels; model_id++ ))
do
    echo "python experiment.py --config $configFile --method $method --model_id $model_id"
done
