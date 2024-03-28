from lnl_surrogate import train


if __name__ == '__main__':
    train(
        model_type='gp',
        compas_h5_filename="../../../data/Z_all/COMPAS_Output.h5",
        params=["aSF", "dSF", "sigma_0", "mu_z"],
        outdir="out_4d_run3",
        acquisition_fns=["pv", "ei", "nlcb"],
        n_init=10,
        n_rounds=4*20,
        n_pts_per_round=10,
        verbose=0
    )