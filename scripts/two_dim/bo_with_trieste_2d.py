""" BO with Trieste"""
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

OUTDIR = "outdir"

# if os.path.exists(OUTDIR):
#     shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(0)

NOISE_LEVEL = 5

N_INIT = 2
N_ROUDNS = 60
N_OPT_STEPS = 1

MOCK_DATA: MockData = generate_mock_data(outdir=OUTDIR)
TRUE = {
    k:MOCK_DATA.observations.mcz_grid.cosmological_parameters[k] for k in ['aSF', 'sigma_0']
}

SF_PRIOR = get_star_formation_prior()
X_RANGE = (
    [SF_PRIOR['aSF'].minimum, SF_PRIOR['sigma0'].minimum],
    [SF_PRIOR['aSF'].maximum, SF_PRIOR['sigma0'].maximum]
)


def f(x):
    if isinstance(x, tf.Tensor):
        x = x.numpy()

    lnls = [
        McZGrid.lnl(
            mcz_obs=MOCK_DATA.observations.mcz,
            duration=1,
            compas_h5_path=MOCK_DATA.compas_filename,
            sf_sample=dict(aSF=_xi[0], sigma0=_xi[1]),
            n_bootstraps=0,
        )[0] * -1 for _xi in x
    ]
    _t = tf.convert_to_tensor(lnls, dtype=tf.float64)
    return tf.reshape(_t, (-1, 1))


def plot_model_and_data(
        model, i, num_initial_points, query_points, search_space,
):
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

    ax[0, 0].scatter(query_points[:, 0], query_points[:, 1], c='black', marker='x', zorder=100)
    ax[0, 1].scatter(query_points[:, 0], query_points[:, 1], c='black', marker='x', zorder=100)

    ax[0, 0].scatter(TRUE['aSF'], TRUE['sigma_0'], marker='*', zorder=1000, color='red')
    ax[0, 1].scatter(TRUE['aSF'], TRUE['sigma_0'], marker='*', zorder=1000, color='red')

    fig.savefig(os.path.join(OUTDIR, f"2d_model_{i}.png"))

    fig, ax = plot_gp_2d(
        model,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        contour=False,
        figsize=(10, 6),
        xlabel="$X_1$",
        ylabel="$X_2$",
        predict_y=True
    )
    fig.savefig(os.path.join(OUTDIR, f"3d_model_{i}.png"))



def main():
    print("Init model")
    search_space = trieste.space.Box(X_RANGE[0], X_RANGE[1])
    observer = mk_observer(f)

    initial_query_points = search_space.sample(N_INIT)
    initial_data = observer(initial_query_points)
    _data = initial_data

    gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=NOISE_LEVEL ** 2)
    model = GaussianProcessRegression(gpflow_model)

    summary_writer = tf.summary.create_file_writer(OUTDIR)
    trieste.logging.set_tensorboard_writer(summary_writer)
    # visualise plots/training progress with tensorboard --logdir=outdir

    # Acquisition functions
    _pi = PredictiveVariance()
    _pi_rule = EfficientGlobalOptimization(_pi)
    _ei = ExpectedImprovement()
    _ei_rule = EfficientGlobalOptimization(_ei)
    rules = [_pi_rule, _ei_rule]
    # rules = [_ei_rule]

    bo = BayesianOptimizer(observer, search_space)

    for i in range(N_ROUDNS):
        _rule = rules[i % len(rules)]
        print("Round: ", i, "Rule: ", _rule)

        result = bo.optimize(N_OPT_STEPS, _data, model, _rule, track_state=False)
        _data = result.try_get_final_dataset()
        model = result.try_get_final_model()
        plot_model_and_data(
            model, i, len(initial_query_points), _data.query_points, search_space
        )

    gp_dataset = result.try_get_final_dataset()
    gp_query_points = gp_dataset.query_points.numpy()
    gp_observations = gp_dataset.observations.numpy()
    gp_arg_min_idx = tf.squeeze(tf.argmin(gp_observations, axis=0))

    fig, ax = plt.subplots(1, 1)
    fig = plot_regret(
        gp_observations,
        ax,
        num_init=len(initial_query_points),
        idx_best=gp_arg_min_idx,
    )
    ax.set_xlim(left=2)
    ax.set_xlabel("# Evaluations")
    ax.set_ylabel("Regret")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "regret.png"))


if __name__ == "__main__":
    main()
