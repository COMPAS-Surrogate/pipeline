#!/bin/bash
#
#SBATCH --job-name=lvk50
#SBATCH --output=logs/lvk%j.out
#SBATCH --error=log/lvk%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=40G
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1

ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate

readarray -t CMDS < 'lvk_bash.sh'

echo "<<<<<< RUNNING cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]} >>>>>>"
eval "${CMDS[$SLURM_ARRAY_TASK_ID]}"
