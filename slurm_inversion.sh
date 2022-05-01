#!/bin/bash

numberOfRepetition=50
configFile=""
method="inversion"
#SBATCH -J (configFile)_inversion
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=1-1
#SBATCH -o slurm_log/%x_%j_%a.log
#SBATCH -e slurm_log/%x_%j_%a.err

for (( repetition_id = 0; repetition_id < $numberOfRepetition; repetition_id++ ))
do
    echo "python experiment.py --config $configFile --method $method --repetition_id $repetition_id"
done
