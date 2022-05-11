#!/bin/bash
#SBATCH -J tandem_inverse_design
#SBATCH -p gpu20
#SBATCH -t 15:00
#SBATCH --gres gpu:1
#SBATCH --array=0-0
#SBATCH -o /HPS/uana/work/slurm_log/%x-%j-%a.log
#SBATCH -e /HPS/uana/work/slurm_log/%x-%j-%a.err

configFile=""
experiment_repetition_id=0
method="tandem"
step="inverse_design"

python main.py --config $configFile --method $method --step $step --experiment_repetition_id $experiment_repetition_id
