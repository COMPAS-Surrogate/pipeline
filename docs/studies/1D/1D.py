# %%
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


def plot_res(model, data, search_space: SearchSpace):
    x = np.linspace(search_space.lower, search_space.upper, 100).reshape(-1, 1)
    # true_y = NORM.logpdf(x) * -1.0
    model_y, model_yunc = model.predict(x)
    x_obs = data.query_points
    y_obs = data.observations

    tf_to_np = lambda x: x.numpy().flatten() if hasattr(x, 'numpy') else x
    # make new fig
    plt.figure()
    # plt.plot(x, true_y, label='True', color='black')
    plt.plot(x, model_y, label='Model', color="tab:orange")
    plt.scatter(x_obs, y_obs, label='Observed', color='black')
    yup, ydown = tf_to_np(model_y + model_yunc), tf_to_np(model_y - model_yunc)
    plt.fill_between(x.flatten(), yup.flatten(), ydown.flatten(), alpha=0.2, color="tab:orange")
    plt.legend(loc='upper right')
    return plt.gcf()


mock_data = generate_mock_data(OUTDIR)


for param in [ 'mu_z', 'sigma_0', 'aSF', 'dSF',]:
    kwargs = dict(
        model_type='gp',
        mcz_obs=mock_data.observations.mcz,
        compas_h5_filename=mock_data.compas_filename,
        params=[param],
        n_init=2,
        n_rounds=10,
        n_pts_per_round=1,
        model_plotter=plot_res,
        noise_level=1e-3,
        truth={
            param: mock_data.truth[param],
            "lnl": mock_data.truth['lnl']
        }
    )
    res = train(
        **kwargs,
        acquisition_fns=[PredictiveVariance(), ExpectedImprovement()],
        outdir=f"{OUTDIR}/gp_{param}/both",
    )

