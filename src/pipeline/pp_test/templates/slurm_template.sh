#SBATCH --job-name=pp_test
#SBATCH --output={{LOG}}/job_%A_%a.out
#SBATCH --error={{LOG}}/job_%A_%a.err
#SBATCH --time=00:100:00
#SBATCH --array=0-{NJOBS}
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1


read_list_from_file() {
    local my_array=()
    while IFS= read -r line; do
        my_array+=("$line")
    done < "$1" # Specify the file to read
}

gen_cmd=($(read_list_from_file {{GEN_CMD_FILE}}))
analy_cmd=($(read_list_from_file {{ANALY_CMD_FILE}}))

echo "Mock cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
echo "Surrogate cmd: ${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
srun ${gen_cmd[$SLURM_ARRAY_TASK_ID]}
srun ${analy_cmd[$SLURM_ARRAY_TASK_ID]}



