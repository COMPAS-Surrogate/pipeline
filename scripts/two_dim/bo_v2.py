from lnl_computer.observation.mock_observation import MockObservation
from lnl_computer.mock_data import MockData, generate_mock_data
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior
import tensorflow as tf
from lnl_surrogate.active_learner import train_and_save_lnl_surrogate
import matplotlib.pyplot as plt

import trieste
from trieste.objectives import Branin, mk_observer
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.experimental.plotting import plot_regret, plot_gp_2d, plot_acq_function_2d
from trieste.experimental.plotting import plot_bo_points, plot_function_2d
import tensorflow as tf
import shutil
import os

from lnl_computer.observation.mock_observation import MockObservation
from lnl_computer.mock_data import MockData, generate_mock_data
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior

import numpy as np

OUTDIR = 'out_test'
MOCK_DATA: MockData = generate_mock_data(outdir=OUTDIR)
TRUE = {
    k: MOCK_DATA.observations.mcz_grid.cosmological_parameters[k] for k in ['aSF', 'sigma_0']
}
TRUE_LNL = McZGrid.lnl(
    mcz_obs=MOCK_DATA.observations.mcz,
    duration=1,
    compas_h5_path=MOCK_DATA.compas_filename,
    sf_sample=TRUE,
    n_bootstraps=0,
)[0] *-1
_pi = PredictiveVariance()
_ei = ExpectedImprovement()

N_INIT = 2
N_ROUDNS = 4
N_OPT_STEPS = 10


def main(outdir=OUTDIR, acquisition_fns=[_pi, _ei]):
    result = train_and_save_lnl_surrogate(
        model_type='gp',
        mcz_obs=MOCK_DATA.observations.mcz,
        compas_h5_filename=MOCK_DATA.compas_filename,
        params=['aSF', 'sigma0'],
        acquisition_fns=acquisition_fns,
        n_init=N_INIT,
        n_rounds=N_ROUDNS,
        n_pts_per_round=N_OPT_STEPS,
        outdir=outdir,
        model_plotter=plot_model_and_data
    )

    gp_dataset = result.try_get_final_dataset()
    gp_query_points = gp_dataset.query_points.numpy()
    gp_observations = gp_dataset.observations.numpy()
    gp_arg_min_idx = tf.squeeze(tf.argmin(gp_observations, axis=0))

    fig, ax = plt.subplots(1, 1)
    fig = plot_regret(
        gp_observations,
        ax,
        num_init=N_INIT,
        idx_best=gp_arg_min_idx,
    )
    ax.set_xlim(left=N_INIT)
    ax.set_ylim(bottom=TRUE_LNL - 5000, top=-50000)
    ax.axhline(TRUE_LNL, color="tab:red")
    ax.set_xlabel("# Evaluations")
    ax.set_ylabel("Regret")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "regret.png"))
    return gp_observations


def plot_model_and_data(
        model, data, search_space
):
    data = data.query_points
    fig, ax = plot_gp_2d(
        model,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        contour=True,
        figsize=(10, 6),
        xlabel="$X_1$",
        ylabel="$X_2$",
        predict_y=True
    )

    ax[0, 0].scatter(data[:, 0], data[:, 1], c='black', marker='x', zorder=100)
    ax[0, 1].scatter(data[:, 0], data[:, 1], c='black', marker='x', zorder=100)

    ax[0, 0].scatter(TRUE['aSF'], TRUE['sigma_0'], marker='*', zorder=1000, color='red')
    ax[0, 1].scatter(TRUE['aSF'], TRUE['sigma_0'], marker='*', zorder=1000, color='red')

    return fig


pi_obs = main('out_pi', acquisition_fns=[_pi])
ei_obs = main('out_ei', acquisition_fns=[_ei])
both_obs = main('out_both', acquisition_fns=[_pi, _ei])

fig, ax = plt.subplots(1, 1)
max_accum = -100000000
for i, (obs, label) in enumerate(zip([pi_obs, ei_obs, both_obs], ['pi', 'ei', 'both'])):

    accum = np.minimum.accumulate(obs)
    # skip n-init pts
    accum = accum[N_INIT:]
    pts = np.arange(N_INIT, N_INIT + len(accum))
    ax.plot(pts, accum, color=f"C{i}", label=label)

    if np.max(accum) > max_accum:
        max_accum = np.max(accum)

ax.set_ylim(bottom=TRUE_LNL - 5000, top=max_accum)
ax.axhline(TRUE_LNL, color="tab:red")
ax.set_xlabel("# Evaluations")
ax.set_ylabel("Regret")
ax.legend()
plt.tight_layout()
plt.savefig("regret.png")
