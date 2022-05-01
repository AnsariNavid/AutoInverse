#!/bin/bash

numberOfRepetition=50
configFile=""
method="inversion"
#SBATCH -J (configFile)_inversion
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=1-50
#SBATCH -o slurm_log/slurm-%x-%j-%a.log
#SBATCH -e slurm_log/slurm-%x-%j-%a.err

currentJob=1
for (( repetition_id = 0; repetition_id < $numberOfRepetition; repetition_id++ ))
do
    if [[ $currentJob -eq $SLURM_ARRAY_TASK_ID ]]
    then
        python experiment.py --config $configFile --method $method --repetition_id $repetition_id
    fi
    currentJob=$currentJob+1
done
