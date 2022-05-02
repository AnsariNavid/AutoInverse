#!/bin/bash
#SBATCH -J train_mu_model
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=0-9
#SBATCH -o /HPS/uana/work/slurm_log/%x-%j-%a.log
#SBATCH -e /HPS/uana/work/slurm_log/%x-%j-%a.err

numberOfModels=10
configFile=""
method="train_mu_model"
currentJob=0
for (( model_id = 0; model_id < $numberOfModels; model_id++ ))
do
    if [[ $currentJob -eq $SLURM_ARRAY_TASK_ID ]]
    then
        python experiment.py --config $configFile --method $method --model_id $model_id
    fi
    currentJob=$currentJob+1
done
