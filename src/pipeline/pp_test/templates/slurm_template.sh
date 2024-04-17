#!/bin/bash
#
#SBATCH --job-name=pp_test
#SBATCH --output={{LOG}}/job_%A_%a.out
#SBATCH --error={{LOG}}/job_%A_%a.err
#SBATCH --time=00:100:00
#SBATCH --array=0-{{NJOBS}}
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1


readarray -t gen_cmd < '{{GEN_CMD_FILE}}'
readarray -t analy_cmd < '{{ANALY_CMD_FILE}}'

echo "Mock cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
echo "Surrogate cmd: ${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
srun ${gen_cmd[$SLURM_ARRAY_TASK_ID]}
srun ${analy_cmd[$SLURM_ARRAY_TASK_ID]}



