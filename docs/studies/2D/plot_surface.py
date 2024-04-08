import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.space import SearchSpace, Box

from lnl_surrogate import LnLSurrogate
import glob
from lnl_computer.cosmic_integration.star_formation_paramters import STAR_FORMATION_RANGES
from lnl_surrogate.plotting.image_utils import make_gif

np.random.seed(0)

OUTDIR = 'outdir/gp_mu_z_sigma_0_run28mar'
os.makedirs(OUTDIR, exist_ok=True)


def plot_res(model, data, search_space: SearchSpace, **kwargs):
    # grid of samples
    n = 100
    x_range = [search_space.lower[0], search_space.upper[0]]
    y_range = [search_space.lower[1], search_space.upper[1]]
    x = np.linspace(*x_range, n)
    y = np.linspace(*y_range, n)
    xx, yy = np.meshgrid(x, y)
    # to tf tensor
    samples = np.c_[xx.flatten(), yy.flatten()]
    samples = tf.constant(samples, dtype=tf.float64)

    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, 'numpy') else x
    model_out, model_unc = model.predict(samples)
    x_obs = tf_to_np(data.mu_z)
    y_obs = tf_to_np(data.sigma_0)
    z_obs = tf_to_np(data.lnl)

    model_lnl = tf_to_np(model_out)
    model_unc = tf_to_np(model_unc)
    # normalise so that the minimum is 0.0001
    model_lnl -= model_lnl.min()
    model_lnl += 0.0001

    # plot of samples along x-y and model_out as z using log scale
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(model_lnl.reshape(n, n), extent=[*x_range, *y_range], origin='lower', aspect='auto', cmap='viridis',
                   norm='log')
    axes[0].scatter(x_obs, y_obs, c=z_obs, edgecolors='white', norm='log')
    axes[0].set_title('Model Mean')

    axes[1].imshow(model_unc.reshape(n, n), extent=[*x_range, *y_range], origin='lower', aspect='auto', cmap='viridis',
                   norm='log')
    axes[1].scatter(x_obs, y_obs, c='w', edgecolors='white', norm='log')
    axes[1].set_title('Model Uncertainty')

    tru_lnzs = np.load(f"{OUTDIR}/truth.npz")
    z = tru_lnzs['z']
    x_range = [tru_lnzs['x'].min(), tru_lnzs['x'].max()]
    y_range = [tru_lnzs['y'].min(), tru_lnzs['y'].max()]
    axes[2].imshow(z.reshape(25, 25), extent=[*x_range, *y_range], origin='lower', aspect='auto', cmap='viridis',
                   norm='log')
    axes[2].set_title('Truth')

    if 'truth' in kwargs:
        truth = kwargs['truth'].copy()
        if 'lnl' in truth:
            # drop lnl
            truth.pop('lnl')
        parm_names = list(truth.keys())
        plt.scatter(truth[parm_names[0]], truth[parm_names[1]], c='tab:orange', marker='s', s=20)
        plt.axvline(truth[parm_names[0]], c='tab:orange', linestyle='-')
        plt.axhline(truth[parm_names[1]], c='tab:orange', linestyle='-')
        # add axes labels
        plt.xlabel(parm_names[0])
        plt.ylabel(parm_names[1])

    return fig


search_space = Box(
    [STAR_FORMATION_RANGES['mu_z'][0], STAR_FORMATION_RANGES['sigma_0'][0]],
    [STAR_FORMATION_RANGES['mu_z'][1], STAR_FORMATION_RANGES['sigma_0'][1]],
)
model_paths = glob.glob(f"{OUTDIR}/round*_*pts")
# sort by pts

plot_out = f"{OUTDIR}/plots_surface"
os.makedirs(plot_out, exist_ok=True)

img_pths = []
model_paths.sort(key=lambda x: int(x.split('_')[-1].split('pts')[0]))
for model_path in model_paths:
    lnl_surr = LnLSurrogate.load(model_path)
    data = lnl_surr.data
    plot_res(lnl_surr.model, data, search_space, truth=lnl_surr.truths)
    fname = f"{plot_out}/{model_path.split('_')[-1]}.png"
    plt.savefig(fname)
    img_pths.append(fname)
    plt.close()
make_gif(image_paths=img_pths, savefn=f"{OUTDIR}/surface.gif")
