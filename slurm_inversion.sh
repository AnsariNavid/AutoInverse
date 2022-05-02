#!/bin/bash
#SBATCH -J <configFile>_inversion
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=0-49
#SBATCH -o /HPS/uana/work/slurm_log/%x-%j-%a.log
#SBATCH -e /HPS/uana/work/slurm_log/%x-%j-%a.err

numberOfRepetition=50
configFile=""
method="inversion"
currentJob=0
for (( repetition_id = 0; repetition_id < $numberOfRepetition; repetition_id++ ))
do
    if [[ $currentJob -eq $SLURM_ARRAY_TASK_ID ]]
    then
        python experiment.py --config $configFile --method $method --repetition_id $repetition_id
    fi
    currentJob=$currentJob+1
done
