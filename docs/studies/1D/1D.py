from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement

import numpy as np
from lnl_surrogate.surrogate import train
from lnl_computer.mock_data import generate_mock_data
import matplotlib.pyplot as plt
import os
from trieste.space import SearchSpace
from lnl_computer.cosmic_integration.star_formation_paramters import STAR_FORMATION_RANGES

np.random.seed(0)

OUTDIR = 'outdir'
os.makedirs(OUTDIR, exist_ok=True)


def plot_res(model, data, search_space: SearchSpace, **kwargs):
    x = np.linspace(search_space.lower, search_space.upper, 100).reshape(-1, 1)
    # true_y = NORM.logpdf(x) * -1.0
    model_y, model_yunc = model.predict(x)
    x_obs = data.query_points
    y_obs = data.observations

    truths = kwargs['reference_param']
    # get keys that arnt lnl
    parm_names = list(truths.keys())
    parm_names.remove('lnl')
    param = parm_names[0]
    true_val = truths[param]


    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, 'numpy') else x
    # make new fig
    plt.figure()
    # plt.plot(x, true_y, label='True', color='black')
    plt.plot(x, model_y, label='Model', color="tab:orange")
    plt.scatter(x_obs, y_obs, label='Observed', color='black')
    yup, ydown = tf_to_np(model_y + model_yunc), tf_to_np(model_y - model_yunc)
    plt.fill_between(x.flatten(), yup.flatten(), ydown.flatten(), alpha=0.2, color="tab:orange")
    plt.axvline(true_val, color='tab:orange', linestyle='--', label='Truth')
    plt.xlabel(param)
    plt.legend(loc='upper right')
    return plt.gcf()


mock_data = generate_mock_data(OUTDIR, duration=2)


for param in [ 'mu_z', 'sigma_0', 'aSF', 'dSF',]:
    kwargs = dict(
        model_type='gp',
        compas_h5_filename="../../../data/Z_all/COMPAS_Output.h5",
        params=[param],
        n_init=2,
        n_rounds=5,
        n_pts_per_round=5,
        model_plotter=plot_res,
        verbose=0
    )
    res = train(
        **kwargs,
        acquisition_fns=["pv", "ei"],
        outdir=f"{OUTDIR}/gp_{param}/both",

    )

