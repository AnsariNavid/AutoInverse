#!/bin/bash
#SBATCH -J uana_train_mu_model
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=0-9
#SBATCH -o /HPS/uana/work/slurm_log/%x-%j-%a.log
#SBATCH -e /HPS/uana/work/slurm_log/%x-%j-%a.err

configFile=""
method="uana"
step="train_mu_model"
numberOfModels=10
currentJob=0
for (( model_id = 0; model_id < $numberOfModels; model_id++ ))
do
    if [[ $currentJob -eq $SLURM_ARRAY_TASK_ID ]]
    then
        python main.py --config $configFile --method $method --step $step --model_id $model_id
    fi
    currentJob=$currentJob+1
done
