from lnl_surrogate.surrogate.train import train

"""
train_lnl_surrogate --compas_h5_filename --mcz_obs_filename  -o /Users/avaj0001/Documents/projects/compas_dev/COSMIC_INT/pp_test/studies/one_param/pp_test/out_aSF/out_surr_0 --n_init 2 --n_rounds 2 --n_pts_per_round 2 --save_plots --reference_param  --duration 2.0 -a pv -a ei -p aSF
"""

train(
    compas_h5_filename='/Users/avaj0001/Documents/projects/compas_dev/COSMIC_INT/pp_test/data/mock_compas.h5',
    mcz_obs_filename='/Users/avaj0001/Documents/projects/compas_dev/COSMIC_INT/pp_test/studies/one_param/pp_test/out_aSF/out_surr_0/mock_obs.npz',
    params=['aSF'],
    model_type="gp",
    duration=2,
    outdir='/Users/avaj0001/Documents/projects/compas_dev/COSMIC_INT/pp_test/studies/one_param/pp_test/out_aSF/out_surr_0',
    acquisition_fns=['pv'],
    n_init=2,
    n_rounds=2,
    n_pts_per_round=1,
    save_plots=True,
    truth='/Users/avaj0001/Documents/projects/compas_dev/COSMIC_INT/pp_test/studies/one_param/pp_test/out_aSF/out_surr_0/reference_param.json',
)
