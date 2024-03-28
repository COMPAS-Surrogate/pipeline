train_lnl_surrogate \
--compas_h5_filename ../../../data/Z_all/COMPAS_Output.h5 \
-p aSF \
--outdir out_1d \
-a pv -a ei -a ei -a ei \
--n_init 5 \
--n_rounds 4 \
--n_pts_per_round 3 \
--save_plots True \