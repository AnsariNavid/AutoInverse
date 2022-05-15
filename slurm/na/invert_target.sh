#!/bin/bash
#SBATCH -J na_invert_target
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=0-249
#SBATCH -o /HPS/uana/work/slurm_log/%x-%j-%a.log
#SBATCH -e /HPS/uana/work/slurm_log/%x-%j-%a.err

configFile=""
experiment_repetition_id=0
method="na"
step="invert_target"
numberOfInversionRepetition=250
currentJob=0
for (( inversion_repetition_id = 0; inversion_repetition_id < $numberOfInversionRepetition; inversion_repetition_id++ ))
do
    if [[ $currentJob -eq $SLURM_ARRAY_TASK_ID ]]
    then
        python main.py --config $configFile --method $method --step $step --experiment_repetition_id $experiment_repetition_id --inversion_repetition_id $inversion_repetition_id
    fi
    currentJob=$currentJob+1
done
