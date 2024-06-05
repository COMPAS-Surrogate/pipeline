"""Given a set of training points, rebuild the LnL surrogate"""
from lnl_surrogate.surrogate import LnLSurrogate
from lnl_surrogate.surrogate.sample import run_sampler
from lnl_surrogate.plotting import plot_overlaid_corner


def rebuild_lnl_surrogate(training_csv: str, outdir: str, label: str = None, model_type: str = 'gp', variable_lnl=False):
    lnl_surr = LnLSurrogate.from_csv(
        training_csv, model_type=model_type, label=label, variable_lnl=variable_lnl
    )
    lnl_surr.plot(outdir=outdir)

    thresh_label = label + "_thresholded" if label is not None else "thresholded"
    thres_lnl_surr = LnLSurrogate.from_csv(
        training_csv, model_type=model_type, label=thresh_label, lnl_threshold=25,
        variable_lnl=variable_lnl
    )
    thres_lnl_surr.plot(outdir=outdir, label=thresh_label)

    res_orig = run_sampler(lnl_surr, outdir=outdir, label=label)
    res_thresh = run_sampler(thres_lnl_surr, outdir=outdir, label=thresh_label, mcmc_kwargs=dict(color="tab:green"))

    truths = {k: lnl_surr.truths[k] for k in lnl_surr.param_keys}

    plot_overlaid_corner(
        [res_thresh.posterior, res_orig.posterior],
        sample_labels=["Thresholded", "Original"],
        axis_labels=lnl_surr.param_latex,
        colors=["tab:purple", "tab:blue"],
        fname=f"{outdir}/corner.png",
        truths=truths,
        label=f"#pts: {lnl_surr.n_training_points}->{thres_lnl_surr.n_training_points}",
    )


if __name__ == "__main__":
    import sys
    cli_args = sys.argv[1:]
    rebuild_lnl_surrogate(*cli_args, variable_lnl=True)