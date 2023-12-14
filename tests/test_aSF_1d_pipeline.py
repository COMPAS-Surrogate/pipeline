import os
import subprocess

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import \
    generate_mock_bbh_population_file
from lnl_computer.cli.main import (batch_lnl_generation, combine_lnl_data,
                                   make_mock_obs, make_sf_table)
from lnl_surrogate.models.sklearn_gp_model import SklearnGPModel
from scipy.optimize import minimize
from scipy.stats import norm
import shutil

LNL_CSV = "combined_lnl_data.csv"
PARAM_CSV = "parameter_table.csv"
COMPAS_H5 = "mock_COMPAS_output.h5"
MOCK_OBS = "mock_obs.npz"
MAX_PTS = 20

# CLI commands
DRAW_SAMPLES_CMD = "make_sf_table"
MOCK_OBS_CMD = "make_mock_obs"
MOCK_COMPAS_CMD = "make_mock_compas_output"
BATCH_LNL_CMD = "batch_lnl_generation"


def _setup(asf_csv, outdir, init_npts: int):
    """Generate inital aSF values, mock datasets"""
    generate_mock_bbh_population_file(filename=COMPAS_H5)
    # subprocess.call([DRAW_SAMPLES_CMD, "-p", "aSF", "-n", init_npts, "-f", asf_csv])
    make_sf_table(parameters=["aSF"], n=init_npts, fname=asf_csv)
    sf_sample = pd.read_csv(asf_csv).to_dict("records")[0]

    # subprocess.call([MOCK_OBS_CMD, COMPAS_H5, sf_sample, "--fname", f"{outdir}/{MOCK_OBS}"])
    make_mock_obs(
        compas_h5_path=COMPAS_H5, sf_sample=sf_sample, fname=f"{outdir}/{MOCK_OBS}"
    )

    # generate mock compas output


def generate_training_data(asf_csv, outdir):
    """Generate training data for lnl, lnl unc, aSF"""
    lnl_fn = f"{outdir}/{LNL_CSV}"
    # subprocess.call([BATCH_LNL_CMD, f"{outdir}/{MOCK_OBS}", COMPAS_H5, asf_csv, "--n_bootstraps", 1, "--fname", lnl_fn])
    batch_lnl_generation(
        mcz_obs=f"{outdir}/{MOCK_OBS}",
        compas_h5_path=COMPAS_H5,
        parameter_table=asf_csv,
        n_bootstraps=1,
        save_images=False,
        outdir=outdir,
    )
    data = pd.read_csv(lnl_fn)
    data = data.sort_values(by=["aSF"])
    # plot data x = aSF, y = lnl and save to outdir
    ax = data.plot.scatter(x="aSF", y="lnl")
    ax.get_figure().savefig(f"{outdir}/lnl_npts{len(data):00d}.png")
    return data


def update_parameter_table_with_acquisition_function(train_data, outdir):
    # acqistion function updates aSF-csv.. temporarily hardcodeed:
    # subprocess.call([DRAW_SAMPLES_CMD, "-p", "aSF", "-n", 10, "-f", f"{outdir}/{PARAM_CSV}"])
    model = SklearnGPModel()
    model.train(
        inputs=train_data.aSF.values.reshape(-1, 1),
        outputs=train_data.lnl.values.reshape(-1, 1),
    )
    new_asf = sample_next_hyperparameter(
        expected_improvement,
        model,
        train_data.lnl.values,
        greater_is_better=True,
        bounds=np.array([[0, 0.2]]),
        n_restarts=25,
    )
    # save new_asf to asf_csv
    new_asf_df = pd.DataFrame({"aSF": new_asf})
    new_asf_df.to_csv(f"{PARAM_CSV}", index=False)


def expected_improvement(x, model, evaluated_loss, greater_is_better=False, n_params=1):
    """expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = model._model.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide="ignore"):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(
            Z
        ) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(
    acquisition_func,
    model,
    evaluated_loss,
    greater_is_better=True,
    bounds=[0, 0.2],
    n_restarts=25,
):
    """sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)
    ):

        res = minimize(
            fun=acquisition_func,
            x0=starting_point.reshape(1, -1),
            bounds=bounds,
            method="L-BFGS-B",
            args=(model, evaluated_loss, greater_is_better, n_params),
        )

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def test_lnl_pipeline(tmp_path):
    np.random.seed(42)
    outdir = f"{tmp_path}/test_lnl_pipeline"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    init_npts = 5
    _setup(asf_csv=PARAM_CSV, outdir=outdir, init_npts=init_npts)

    npts = 0
    while npts < MAX_PTS:
        data = generate_training_data(asf_csv=PARAM_CSV, outdir=outdir)
        update_parameter_table_with_acquisition_function(train_data=data, outdir=outdir)
        npts = len(data)
