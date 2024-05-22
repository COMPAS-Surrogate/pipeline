"""
1. Parse label
2. Determine number of 'rounds' available
3. For each 'round', load the results (for each type) and plot the PP plot
(type of results 1) normal, 2) high-mcmc, 3) variable
4. Save the PP-plot data (for future plotting)
"""
import glob
import sys
import os
from tqdm.auto import tqdm
from pipeline.pp_test.make_pp_plot import make_pp_plot, cache_pp_data
from bilby.core.result import Result

NORMAL = "round{N}_*pts_result.json"
HIGH_MCMC = "round{N}_*pts_highres_result.json"
VARIABLE = "round{N}_*pts_variable_lnl_result.json"

base_dir = "/fred/oz303/avajpeyi/studies/cosmic_int/simulation_study/4dim/out_pp_4/"


def get_n_rounds(surr_dir):
    s = f"{surr_dir}/out_mcmc/{NORMAL.format(N='*')}"
    files = glob.glob(s)
    rounds = [int(os.path.basename(f).split('round')[1].split('_')[0]) for f in files]
    return max(rounds)


def process_round(surr_dir, round_id, outdir):
    normal_files = glob.glob(f"{surr_dir}/out_mcmc/{NORMAL.format(N=round_id)}")
    high_mcmc_files = glob.glob(f"{surr_dir}/out_mcmc/{HIGH_MCMC.format(N=round_id)}")
    variable_files = glob.glob(f"{surr_dir}/out_mcmc/{VARIABLE.format(N=round_id)}")
    os.makedirs(outdir, exist_ok=True)
    save_pp_for_files(normal_files, f"{outdir}/pp_round{round_id}_normal")
    save_pp_for_files(high_mcmc_files, f"{outdir}/pp_round{round_id}_high_mcmc")
    save_pp_for_files(variable_files, f"{outdir}/pp_round{round_id}_variable")


def save_pp_for_files(files, label):
    results = [Result.from_json(f) for f in tqdm(files)]
    make_pp_plot(
        pp_datafn=f"{label}.pkl",
        results=results,
        filename=f"{label}.png",
    )


def main(base_dir):
    surr_base = f"{base_dir}/out_surr_*"
    mcmc_base = f"{surr_base}/out_mcmc/"
    surr_dirs = glob.glob(surr_base)

    n_rounds = get_n_rounds(surr_dirs[0])
    print(f"Round\t#Normal\t#HighMCMC\t#Variable")
    for round in range(n_rounds):
        normal_files = glob.glob(f"{mcmc_base}/{NORMAL.format(N=round)}")
        high_mcmc_files = glob.glob(f"{mcmc_base}/{HIGH_MCMC.format(N=round)}")
        variable_files = glob.glob(f"{mcmc_base}/{VARIABLE.format(N=round)}")
        print(f"{round}\t{len(normal_files)}\t{len(high_mcmc_files)}\t{len(variable_files)}")

    outdir = f"{base_dir}/pp_plots"
    os.makedirs(outdir, exist_ok=True)
    print("Saving PP plots to: ", outdir)
    for round in range(n_rounds):
        print(f"Processing round {round}...")
        process_round(surr_base, round, outdir)


if __name__ == "__main__":
    main(sys.argv[1])