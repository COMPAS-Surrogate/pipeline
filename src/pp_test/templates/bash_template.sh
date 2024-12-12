#!/bin/bash


readarray -t gen_cmd < '{{GEN_CMD_FILE}}'
readarray -t analy_cmd < '{{ANALY_CMD_FILE}}'

# Loop over the arrays using the length of one of them
for ((SLURM_ARRAY_TASK_ID = 0; SLURM_ARRAY_TASK_ID < ${#gen_cmd[@]}; SLURM_ARRAY_TASK_ID++)); do
    echo "Mock cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
    echo "Surrogate cmd: ${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
    # Execute commands with eval to handle cases where command contains spaces
    eval "${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
    eval "${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
done
