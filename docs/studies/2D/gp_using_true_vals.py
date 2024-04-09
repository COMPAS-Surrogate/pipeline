import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from trieste.data import Dataset
from trieste.space import Box
from lnl_surrogate.plotting.image_utils import make_gif

from lnl_surrogate.surrogate.model import get_model

outdir = "gp_with_true_vals"
os.makedirs(outdir, exist_ok=True)


def load_data(frac=1):
    np.random.seed(0)
    data = np.load('truth_lnls.npz')
    rel_neg_lnls = data['z']
    x, y = data['x'], data['y']
    # drop z==0s
    mask = rel_neg_lnls != 0
    query = np.column_stack([x[mask], y[mask]])
    obs = rel_neg_lnls[mask].reshape(-1, 1)
    space = Box([min(x), min(y)], [max(x), max(y)])

    idx =   np.array([i for i in range(len(obs)) if obs[i] != 0])
    np.random.shuffle(idx)
    query, obs = query[idx], obs[idx]

    if frac < 1:
        n = int(frac * len(obs))
        idx = idx[:n]
        query, obs = query[idx], obs[idx]

    return Dataset(query, obs), space


def plot_data_and_gp(gp_model: Dataset, space: Box, label):
    xrange = [space.lower[0], space.upper[0]]
    yrange = [space.lower[1], space.upper[1]]
    x = np.linspace(*xrange, 100)
    y = np.linspace(*yrange, 100)
    X, Y = np.meshgrid(x, y)
    Z, Zunc = gp_model.predict(np.column_stack([X.ravel(), Y.ravel()]))
    true_data = np.load('truth_lnls.npz')
    xtrue, ytrue = true_data['x'].reshape(25, 25), true_data['y'].reshape(25, 25)
    ztrue = true_data['z'].reshape(25, 25)
    Z = Z.numpy().reshape(X.shape)
    Zunc = Zunc.numpy().reshape(X.shape)
    norm = LogNorm(vmin=np.min(ztrue[ztrue>0]), vmax=np.max(ztrue))
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
    fig.suptitle(label, fontsize=25)
    plt.savefig(f'{outdir}/{label}.png')


data, space = load_data()
for f in np.linspace(0.01, 0.05, 20):
    train_data, _ = load_data(f)
    gp_model = get_model('gp', train_data, space)
    gp_model.optimize(train_data)
    npts = len(train_data.observations)
    plot_data_and_gp(gp_model, space, f'gp_npts{npts:003d}')
for f in np.geomspace(1, 0.05, 100):
    train_data, _ = load_data(f)
    gp_model = get_model('gp', train_data, space)
    gp_model.optimize(train_data)
    npts = len(train_data.observations)
    plot_data_and_gp(gp_model, space, f'gp_npts{npts:003d}')


make_gif(f"{outdir}/*.png", 'gp_with_true_vals.gif')