from lnl_surrogate.surrogate.train import Trainer
import os
import numpy as np
import tensorflow as tf
from tqdm.auto import trange

OUTDIR = 'outdir'

dur = 10

for param_set in [
    ['mu_z', 'sigma_0'],
    # ['aSF', 'dSF'],
    # ['mu_z', 'aSF'],
    # ['mu_z', 'dSF'],
    # ['sigma_0', 'aSF'],
    # ['sigma_0', 'dSF'],
]:
    label = "_".join(param_set)
    outdir = f"{OUTDIR}/gp_{label}_TRUE"
    os.makedirs(outdir, exist_ok=True)
    print(f"STARTING {label}")
    kwargs = dict(
        model_type='gp',
        compas_h5_filename="../../../data/Z_all/COMPAS_Output.h5",
        params=param_set,
        n_init=2,
        n_rounds=10,
        n_pts_per_round=5,
        noise_level=1e-3,
        acquisition_fns=['pv'],
        duration=dur
    )

    trainer = Trainer(**kwargs)
    observer = trainer.opt_mngr.bo._observer
    search_space = trainer.opt_mngr.bo._search_space

    n = 25
    x_range = [search_space.lower[0], search_space.upper[0]]
    y_range = [search_space.lower[1], search_space.upper[1]]
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    xx, yy = np.meshgrid(x, y)
    # to tf tensor
    samples = np.c_[xx.flatten(), yy.flatten()]
    tru_lnls = np.zeros(len(samples))
    for i in trange(samples.shape[0]):
        obs = observer(samples[i].reshape(1,-1))
        tru_lnls[i] = obs.observations.numpy().flatten()[0]
        if i == 1:
            np.savez(f"{outdir}/truth_{i}_dur{dur}.npz", x=samples[:, 0], y=samples[:, 1], z=tru_lnls)

        # cache x, y, z every 20 iterations
        if i>0 and i % 20 == 0:
            np.savez(f"{outdir}/truth_{i}_dur{dur}.npz",  x=samples[:,0], y=samples[:,1], z=tru_lnls)
    np.savez(f"{outdir}/truth_dur{dur}.npz", x=samples[:,0], y=samples[:,1], z=tru_lnls)

