#!/bin/bash
#
#SBATCH --job-name=pp_test
#SBATCH --output={{LOG}}/job_%A_%a.out
#SBATCH --error={{LOG}}/job_%A_%a.err
#SBATCH --time={{TIME}}
#SBATCH --array=0-{{NJOBS}}
#SBATCH --mem={{MEM}}
#SBATCH --cpus-per-task=1

ml gcc/11.2.0 python/3.9.6 && source /fred/oz303/avajpeyi/venvs/compas_env/bin/activate

readarray -t gen_cmd < '{{GEN_CMD_FILE}}'
readarray -t analy_cmd < '{{ANALY_CMD_FILE}}'

echo "Mock cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
echo "Surrogate cmd: ${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
eval "${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
eval "${analy_cmd[$SLURM_ARRAY_TASK_ID]}"



