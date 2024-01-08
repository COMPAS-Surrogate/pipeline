""" BO with Trieste"""
import matplotlib.pyplot as plt

import trieste
from trieste.objectives import Branin, mk_observer
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.experimental.plotting import plot_regret
import tensorflow as tf
import shutil
import os

from lnl_computer.observation.mock_observation import MockObservation
from lnl_computer.mock_data import MockData, generate_mock_data
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior


import numpy as np

OUTDIR = "outdir_ei"

# if os.path.exists(OUTDIR):
#     shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(0)

SCALE = 30
OFFSET = -50000
NOISE_LEVEL = 5


N_ROUDNS = 5
N_OPT_STEPS = 3


MOCK_DATA: MockData = generate_mock_data(outdir=OUTDIR)
SF_PRIOR = get_star_formation_prior()
X_RANGE = (SF_PRIOR['aSF'].minimum, SF_PRIOR['aSF'].maximum)


def f(x):

    if isinstance(x, tf.Tensor):
        x = x.numpy()

    lnls = [
        McZGrid.lnl(
            mcz_obs=MOCK_DATA.observations.mcz,
            duration=1,
            compas_h5_path=MOCK_DATA.compas_filename,
            sf_sample=dict(aSF=_xi[0], dSF=4.70, mu_z=-0.23, sigma_z=0.0),
            n_bootstraps=0,
        )[0] * -1 for _xi in x
    ]
    _t = tf.convert_to_tensor(lnls, dtype=tf.float64)
    return tf.reshape(_t, (-1, 1))


def plot_model_and_data(model, init_query, init_data, new_data=None, full_line=False):
    X_VALS = np.linspace(X_RANGE[0], X_RANGE[1], 50)
    model_mean, model_var = model.predict(tf.reshape(X_VALS, (-1, 1)))
    y_up = model_mean + 2 * np.sqrt(model_var)
    y_down = model_mean - 2 * np.sqrt(model_var)

    ei_fn = ExpectedImprovement().prepare_acquisition_function(model, dataset=init_data)
    pv_fn = PredictiveVariance().prepare_acquisition_function(model, dataset=init_data)
    ei_y = tf.squeeze(ei_fn(tf.reshape(X_VALS, (-1, 1, 1))))
    pv_y = tf.squeeze(pv_fn(tf.reshape(X_VALS, (-1, 1, 1))))
    # normalise between 0 and 1
    ei_y = (ei_y - tf.reduce_min(ei_y)) / (tf.reduce_max(ei_y) - tf.reduce_min(ei_y))
    pv_y = (pv_y - tf.reduce_min(pv_y)) / (tf.reduce_max(pv_y) - tf.reduce_min(pv_y))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    if full_line:
        axes[0].plot(X_VALS, f(X_VALS), color='black')
    axes[0].scatter(init_query, init_data.observations, marker='x', color='black', zorder=10)
    axes[0].plot(X_VALS, model_mean, color='tab:blue')
    axes[0].fill_between(X_VALS, tf.squeeze(y_up), tf.squeeze(y_down), alpha=0.2, color='tab:blue')
    axes[0].set_ylabel("-lnl")
    if new_data is not None:
        axes[0].scatter(new_data.query_points, new_data.observations, marker='o')
    axes[1].plot(X_VALS, ei_y, color='tab:orange', label="Exploitation")
    axes[1].plot(X_VALS, pv_y, color='tab:green', label="Exploration")
    axes[0].set_ylabel("Acquisition Fx")
    axes[1].legend()
    return plt.gcf()


def main():
    search_space = trieste.space.Box([X_RANGE[0]], [X_RANGE[1]])
    observer = mk_observer(f)

    initial_query_points = search_space.sample(2)
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
    rules = [_ei_rule]

    bo = BayesianOptimizer(observer, search_space)

    for i in range(N_ROUDNS):
        _rule = rules[i % len(rules)]
        print("Round: ", i, "Rule: ", _rule)
        plot_model_and_data(model, initial_query_points, initial_data, _data).savefig(
            os.path.join(OUTDIR, f"rnd_{i}.png"))
        result = bo.optimize(N_OPT_STEPS, _data, model, _rule, track_state=False)
        _data = result.try_get_final_dataset()
        model = result.try_get_final_model()

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
    ax.set_xlabel("# Evaluations")
    ax.set_ylabel("Regret")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "regret.png"))


if __name__ == "__main__":
    main()
