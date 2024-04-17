#bin/bash
read_list_from_file() {
    local my_array=()
    while IFS= read -r line; do
        my_array+=("$line")
    done < "$1" # Specify the file to read
}

gen_cmd=($(read_list_from_file "gen_cmd.txt"))
analy_cmd=($(read_list_from_file "analy_cmd.txt"))

for SLURM_ARRAY_TASK_ID in $(seq 0 $((${#gen_cmd[@]} - 1))); do
    echo "Mock cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
    echo "Surrogate cmd: ${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
    srun ${gen_cmd[$SLURM_ARRAY_TASK_ID]}
    srun ${analy_cmd[$SLURM_ARRAY_TASK_ID]}
done



