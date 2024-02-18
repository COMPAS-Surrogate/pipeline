import os
import subprocess

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import \
    generate_mock_bbh_population_file
from lnl_computer.cli.main import (batch_lnl_generation, combine_lnl_data,
                                   make_mock_obs, make_sf_table)

from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior
from lnl_surrogate.models.sklearn_gp_model import SklearnGPModel
from lnl_surrogate.active_learner import query_points
from lnl_surrogate.acquisition.ei import expected_improvement
from lnl_surrogate.surrogate import get_star_formation_prior
from scipy.optimize import minimize
from scipy.stats import norm
import shutil
import matplotlib.pyplot as plt

LNL_CSV = "combined_lnl_data.csv"
PARAM_CSV = "parameter_table.csv"
COMPAS_H5 = "mock_COMPAS_output.h5"
MOCK_OBS = "mock_obs.npz"
MAX_PTS = 10

# CLI commands
DRAW_SAMPLES_CMD = "make_sf_table"
MOCK_OBS_CMD = "make_mock_obs"
MOCK_COMPAS_CMD = "make_mock_compas_output"
BATCH_LNL_CMD = "batch_lnl_generation"


def _setup(query_csv, outdir, init_npts: int):
    """Generate inital aSF values, mock datasets"""
    generate_mock_bbh_population_file(filename=COMPAS_H5)
    # subprocess.call([DRAW_SAMPLES_CMD, "-p", "aSF", "-n", init_npts, "-f", asf_csv])
    make_sf_table(parameters=["aSF"], n=init_npts, fname=query_csv)
    sf_sample = pd.read_csv(query_csv).to_dict("records")[0]

    # subprocess.call([MOCK_OBS_CMD, COMPAS_H5, sf_sample, "--fname", f"{outdir}/{MOCK_OBS}"])
    make_mock_obs(
        compas_h5_path=COMPAS_H5, sf_sample=sf_sample, fname=f"{outdir}/{MOCK_OBS}"
    )

    # generate mock compas output


def generate_training_data(query_csv, train_csv, outdir):
    """Generate training data for lnl, lnl unc, aSF"""
    # subprocess.call([BATCH_LNL_CMD, f"{outdir}/{MOCK_OBS}", COMPAS_H5, asf_csv, "--n_bootstraps", 1, "--fname", lnl_fn])
    batch_lnl_generation(
        mcz_obs=f"{outdir}/{MOCK_OBS}",
        compas_h5_path=COMPAS_H5,
        parameter_table=query_csv,
        n_bootstraps=0,
        save_images=False,
        outdir=outdir,
    )


def update_parameter_table_with_acquisition_function(query_csv, train_csv):
    # acqistion function updates aSF-csv.. temporarily hardcodeed:
    # subprocess.call([DRAW_SAMPLES_CMD, "-p", "aSF", "-n", 10, "-f", f"{outdir}/{PARAM_CSV}"])
    print("Querying new aSF values")
    train_data = pd.read_csv(train_csv)
    model = SklearnGPModel()
    model.train(
        inputs=train_data.aSF.values.reshape(-1, 1),
        outputs=train_data.lnl.values.reshape(-1, 1),
    )

    new_asf = query_points(
        trained_model=model,
        training_in=train_data.aSF.values.reshape(-1, 1),
        priors=get_star_formation_prior(parameters=["aSF"]),
        acquisition_function=expected_improvement,
        acquisition_args=[],
    )
    # save new_asf to asf_csv
    new_asf_df = pd.DataFrame({"aSF": new_asf.flatten()})
    # append this to the old asf csv
    old_asf_df = pd.read_csv(query_csv)
    new_asf_df = pd.concat([old_asf_df, new_asf_df])
    print(f"Increasing training size: {len(old_asf_df)}->{len(new_asf_df)}")
    new_asf_df.to_csv(query_csv, index=False)


def plot(train_csv, prior, outdir):
    """Plot the training data, prior, and posterior"""
    train_data = pd.read_csv(train_csv)
    model = SklearnGPModel()
    model.train(
        inputs=train_data.aSF.values.reshape(-1, 1),
        outputs=train_data.lnl.values.reshape(-1, 1),
    )
    x = np.linspace(*prior.bounds[0], 100)
    y, yerr = model._model.predict(x.reshape(-1, 1), return_std=True)
    plt.plot(x, y)
    # plt.scatter(train_data.aSF, train_data.lnl)
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.5)
    plt.savefig(f"{outdir}/lnl_{len(train_data):002d}.png")


def test_lnl_pipeline(tmp_path):
    np.random.seed(42)
    outdir = f"{tmp_path}/test_lnl_pipeline"
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    query_csv = f"{outdir}/{PARAM_CSV}"
    train_csv = f"{outdir}/{LNL_CSV}"


    init_npts = 2
    _setup(query_csv=query_csv, outdir=outdir, init_npts=init_npts)

    npts = 0
    while npts < MAX_PTS:
        generate_training_data(train_csv=train_csv, query_csv=query_csv, outdir=outdir)
        update_parameter_table_with_acquisition_function(query_csv=query_csv, train_csv=train_csv)
        npts = len(pd.read_csv(train_csv))
        plot(train_csv=train_csv, prior=get_star_formation_prior(parameters=["aSF"]), outdir=outdir)
