from lnl_computer.observation.mock_observation import MockObservation
from lnl_computer.mock_data import MockData, generate_mock_data
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior
import tensorflow as tf


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
from lnl_surrogate.surrogate import train

import numpy as np

PARAMS = ['aSF', 'dSF', 'mu_z', 'sigma_0']

OUTDIR = 'out_test'
MOCK_DATA: MockData = generate_mock_data(outdir=OUTDIR)
TRUE = {
    k: MOCK_DATA.observations.mcz_grid.cosmological_parameters[k] for k in PARAMS
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

N_INIT = 10
N_ROUDNS = 4
N_OPT_STEPS = 10


def main(outdir=OUTDIR, acquisition_fns=[_pi, _ei]):
    result = train(
        model_type='gp',
        mcz_obs=MOCK_DATA.observations.mcz,
        compas_h5_filename=MOCK_DATA.compas_filename,
        params=PARAMS,
        acquisition_fns=acquisition_fns,
        n_init=N_INIT,
        n_rounds=N_ROUDNS,
        n_pts_per_round=N_OPT_STEPS,
        outdir=outdir,
        model_plotter=None,

        truth={
            **TRUE,
            "lnl": TRUE_LNL
        }
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
    # ax.set_ylim(bottom=TRUE_LNL - 5000, top=-50000)
    ax.axhline(TRUE_LNL, color="tab:red")
    ax.set_xlabel("# Evaluations")
    ax.set_ylabel("Regret")
    np.savetxt(os.path.join(outdir, "y.txt"), gp_observations)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "regret.png"))
    return gp_observations



both_obs = main('out_both', acquisition_fns=[_pi, _ei])
label = 'explore+exploit'

fig, ax = plt.subplots(1, 1)
# max_accum = -100000000

accum = np.minimum.accumulate(both_obs)
# skip n-init pts
accum = accum[N_INIT:]
pts = np.arange(N_INIT, N_INIT + len(accum))
ax.plot(pts, accum, color=f"C0", label=label)


# ax.set_ylim(bottom=TRUE_LNL - 5000, top=max_accum)
ax.axhline(TRUE_LNL, color="tab:red")
ax.set_xlabel("# Evaluations")
ax.set_ylabel("Regret")
ax.legend()
plt.tight_layout()
plt.savefig("regret.png")
