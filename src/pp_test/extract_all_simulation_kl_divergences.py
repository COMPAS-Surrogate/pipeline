import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lnl_surrogate.kl_distance.batch_distance_computer import save_distances, _get_result_fnames, _get_npts_from_fname


def save_distances_for_entire_result_dir(base_dir: str, reference_posterior_round_num: int = 33):
    """Run the distance computation for all results in a directory"""
    result_dirs = glob.glob(os.path.join(base_dir, "out_surr_*/out_mcmc"))
    for res_dir in tqdm(result_dirs, desc="Processing result directories"):
        ref_res, high_round_num = _get_highest_res_result_fname(res_dir)
        if high_round_num > reference_posterior_round_num:  # For dirs with more than 33 rounds -- use largest round as reference (probably 33rd)
            logger.info(
                f"Getting KLs for all results with {os.path.join(res_dir, '*variable_lnl_result.json')} + ref: {ref_res_fname}")
            save_distances(
                res_regex=os.path.join(res_dir, "*variable_lnl_result.json"),
                ref_res_fname=ref_res_fname,
                fname=os.path.join(res_dir, "variable_kldists_hist.csv")
            )
    logger.info("\n\n------Done!!!------")


def load_all_kl_datasets(base_dir: str):
    """Load all the KL datasets from a directory"""
    kl_distance_csv = glob.glob(os.path.join(base_dir, "out_surr_*/out_mcmc/variable_kldists_hist.csv"))
    dfs = [pd.read_csv(f) for f in kl_distance_csv]
    max_len = max(len(df) for df in dfs)
    dfs = [df for df in dfs if len(df) == max_len]
    data = {f"{stat}_{i}": np.pad(df[stat], (0, max_len - len(df)), 'constant')
            for i, df in enumerate(dfs)
            for stat in ['kl', 'ks', 'js']}
    data['npts'] = dfs[np.argmax([len(df) for df in dfs])]['npts']
    df = pd.DataFrame(data)
    return df


def pre_process_all_kl_data(df, smooth_factor=0.00005, metric='kl_distance'):
    # preprocessing data
    x = df['npts']
    new_x = np.arange(df['npts'].min(), df['npts'].max(), 10)
    new_x = np.unique(np.round(new_x).astype(int))
    metric_data = df.filter(like=metric).values
    _, n_data = metric_data.shape
    medians = _smooth(x, np.median(metric_data, axis=1), new_x, s=smooth_factor, k=3)
    lower_ci = _smooth(x, np.percentile(metric_data, 10, axis=1), new_x, s=smooth_factor / 10, k=2)
    upper_ci = _smooth(x, np.percentile(metric_data, 80, axis=1), new_x, s=smooth_factor * 150, k=1)
    ci_data = pd.DataFrame(dict(
        npts=new_x,
        lower_ci_95=lower_ci,
        upper_ci_95=upper_ci,
        medians=medians
    ))
    ci_data.to_csv(f"{metric}_ci_data.csv", index=False)


def _smooth(x, y, new_x, s=0.005, k=1):
    return UnivariateSpline(x, y, s=s, k=k)(new_x)


def plot_smoothed_metrics(metric='kl'):
    data = pd.read_csv(f"{metric}_ci_data.csv")
    x, y, yu, yl = data.npts, data.medians, data.upper_ci_95, data.lower_ci_95
    x, y, yu, yl = np.array(x.tolist()), y.tolist(), yu.tolist(), yl.tolist()
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    axs.plot(x, y, label='Median')
    axs.fill_between(x.tolist(), yl, yu, alpha=0.3, label='95% CI')
    axs.set_ylabel(f'{metric.capitalize().replace("_", " ")}')
    axs.set_xlabel('Number of GP training Points')
    axs.legend()
    axs.set_xlim(min(data.npts), 900)
    plt.tight_layout()
    axs.set_ylim(bottom=0)
    fig.savefig(f"{metric}.png")



def extract_all_simulation_kl_divergences(base_dir:str, metric:str):
    """
    Run the entire pipeline
    Args:
    base_dir: str
        The base directory containing the results
    metric: str
        The metric to plot (kl, ks, js)
    """
    save_distances_for_entire_result_dir(base_dir)
    df = load_all_kl_datasets(base_dir)
    pre_process_all_kl_data(df, metric=metric)
    plot_smoothed_metrics(metric)

