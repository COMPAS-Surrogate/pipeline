import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.acquisition import ExpectedImprovement, PredictiveVariance, NegativeLowerConfidenceBound
from trieste.space import SearchSpace
from matplotlib.colors import LogNorm
from lnl_surrogate import train
from trieste.data import Dataset

np.random.seed(0)

OUTDIR = 'outdir/'
os.makedirs(OUTDIR, exist_ok=True)


def plot_data_and_gp(model, data:Dataset, search_space: SearchSpace, **kwargs):
    xrange = [search_space.lower[0], search_space.upper[0]]
    yrange = [search_space.lower[1], search_space.upper[1]]
    x = np.linspace(*xrange, 100)
    y = np.linspace(*yrange, 100)
    X, Y = np.meshgrid(x, y)
    Z, Zunc = model.predict(np.column_stack([X.ravel(), Y.ravel()]))
    true_data = np.load('truth_lnls.npz')
    xtrue, ytrue = true_data['x'].reshape(25, 25), true_data['y'].reshape(25, 25)
    ztrue = true_data['z'].reshape(25, 25)
    Z = Z.numpy().reshape(X.shape)
    Zunc = Zunc.numpy().reshape(X.shape)
    norm = LogNorm(vmin=np.min(ztrue[ztrue > 0]), vmax=np.max(ztrue))
    my_dpi = 300
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=my_dpi)
    kwgs = dict(cmap='viridis', aspect='equal', interpolation='none', extent=[*xrange, *yrange], origin='lower',
                norm=norm)
    im = axes[0].imshow(Zunc, **kwgs)
    im = axes[1].imshow(Z, **kwgs)
    im = axes[2].pcolormesh(xtrue, ytrue, ztrue, norm=norm, edgecolors='w', lw=my_dpi / (1024 * 32), shading='auto')
    axes[0].set_title('Surrogate Uncertainty', fontsize=20)
    axes[1].set_title('Surrogate Mean', fontsize=20)
    axes[2].set_title('True', fontsize=20)
    axes[2].set_aspect('equal')
    for ax in axes:
        ax.set_ylim(top=0.58)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    # colorbar axes label ylabel
    cbar.set_label('Rel Neg lnL', fontsize=20)

    fig.suptitle(f"N={len(data.observations)}", fontsize=25)

    for ax in [axes[0], axes[1]]:
        # scatter true data values
        ax.scatter(data.query_points[:, 0], data.query_points[:, 1], c='tab:red', marker='.', s=2)

    if 'reference_param' in kwargs:
        truth = kwargs['reference_param'].copy()
        if 'lnl' in truth:
            # drop lnl
            truth.pop('lnl')
        parm_names = list(truth.keys())
        for ax in [axes[1], axes[2]]:
            ax.scatter(truth[parm_names[0]], truth[parm_names[1]], c='tab:orange', marker='s', s=2)
            ax.axvline(truth[parm_names[0]], c='tab:orange', linestyle='-')
            ax.axhline(truth[parm_names[1]], c='tab:orange', linestyle='-')
        # add axes labels
        for ax in axes:
            ax.set_xlabel(parm_names[0])
            ax.set_ylabel(parm_names[1])

    return fig


for param_set in [
    ['mu_z', 'sigma_0'],
    # ['aSF', 'dSF'],
    # ['mu_z', 'aSF'],
    # ['mu_z', 'dSF'],
    # ['sigma_0', 'aSF'],
    # ['sigma_0', 'dSF'],
]:
    label = "_".join(param_set)
    outdir = f"{OUTDIR}/gp_{label}_run28mar"
    print(f"STARTING {label}")

    res = train(
        model_type='gp',
        compas_h5_filename="../../../data/Z_all/COMPAS_Output.h5",
        params=param_set,
        n_init=2,
        n_rounds=100,
        n_pts_per_round=5,
        model_plotter=plot_data_and_gp,
        duration=10,
        acquisition_fns=[
            "pv", "ei", "nlcb"
        ],
        outdir=outdir,
    )
