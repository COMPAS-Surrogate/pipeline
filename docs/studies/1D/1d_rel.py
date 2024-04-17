from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement

import numpy as np
from lnl_surrogate.surrogate import train
from lnl_computer.mock_data import generate_mock_data
import matplotlib.pyplot as plt
import os
from trieste.space import SearchSpace
from lnl_surrogate.surrogate.setup_optimizer import McZGrid
from lnl_computer.mock_data import generate_mock_data

np.random.seed(0)

OUTDIR = 'outdir_rel'
os.makedirs(OUTDIR, exist_ok=True)

TRUE_LNL = -112241.07371310738


def mock_lnl(
        sf_sample, mcz_obs: np.ndarray, duration=1, **kwargs):
    """Return the LnL(sf_sample|mcz_obs)+/-unc

    Also saves the Lnl+/-unc and params to a csv file

    :param mcz_obs: The observed mcz values
    :param duration: The duration of the observation (in years)
    :param args: Arguments to pass to generate_n_save
    :return: The LnL value
    """
    model = McZGrid.generate_n_save(**kwargs, sf_sample=sf_sample)
    lnl, unc = model.get_lnl(mcz_obs=mcz_obs, duration=duration)

    return lnl-TRUE_LNL, unc

McZGrid.lnl = mock_lnl
mock_data = generate_mock_data(OUTDIR)


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



for param in ['dSF']:#, 'muz', 'sigma0', 'aSF']:
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
            "lnl": 0
        }
    )
    res = train(
        **kwargs,
        acquisition_fns=[PredictiveVariance(), ExpectedImprovement()],
        outdir=f"{OUTDIR}/gp_{param}/both",
    )





# from lnl_surrogate.plotting.regret_plots import plot_multiple_regrets, RegretData
# from lnl_computer.mock_data import load_mock_data
#
# mock_data = load_mock_data('outdir')
#
# labels = ['Explore', 'Exploit', 'Both']
# params = ['dSF', 'muz', 'sigma0', 'aSF']
# colors = ['blue', 'orange', 'green']
#
# data_fmt = 'outdir/gp_{param}/{label}/regret.csv'
#
# for param in params:
#     regret_data = [
#         RegretData(data_fmt.format(param=param, label=labels[i].lower()), labels[i], colors[i]) for i in range(3)
#     ]
#     plot_multiple_regrets(
#         regret_data, fname=f'outdir_rel/gp_{param}/regret.png',
#         true_min=mock_data.truth['lnl'],
#     )
#
