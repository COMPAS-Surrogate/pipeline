import subprocess, os
import pandas as pd
import numpy as np

from lnl_computer.cli.main import make_sf_table, batch_lnl_generation, combine_lnl_data, make_mock_obs
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.bbh_population import \
    generate_mock_bbh_population_file

np.random.seed(42)

LNL_CSV = "combined_lnl_data.csv"
PARAM_CSV = "parameter_table.csv"
COMPAS_H5 = "mock_COMPAS_output.h5"
MOCK_OBS = "mock_obs.npz"
MAX_PTS = 50

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
    make_mock_obs(compas_h5_path=COMPAS_H5, sf_sample=sf_sample, fname=f"{outdir}/{MOCK_OBS}")

    # generate mock compas output


def generate_training_data(asf_csv, outdir):
    """Generate training data for lnl, lnl unc, aSF"""
    lnl_fn = f"{outdir}/{LNL_CSV}"
    # subprocess.call([BATCH_LNL_CMD, f"{outdir}/{MOCK_OBS}", COMPAS_H5, asf_csv, "--n_bootstraps", 1, "--fname", lnl_fn])
    batch_lnl_generation(mcz_obs=f"{outdir}/{MOCK_OBS}", compas_h5_path=COMPAS_H5, parameter_table=asf_csv, n_bootstraps=1, save_images=False, outdir=outdir)
    data = pd.read_csv(lnl_fn)
    data = data.sort_values(by=["aSF"])
    # plot data x = aSF, y = lnl and save to outdir
    fig = data.plot(x="aSF", y="lnl")
    fig.savefig(f"{outdir}/lnl_npts{len(data):00d}.png")
    return data


def train_and_evaluate_model(train_data, outdir):
    # acqistion function updates aSF-csv.. temporarily hardcodeed:
    # subprocess.call([DRAW_SAMPLES_CMD, "-p", "aSF", "-n", 10, "-f", f"{outdir}/{PARAM_CSV}"])
    make_sf_table(parameters=["aSF"], n=10, fname=f"{outdir}/{PARAM_CSV}")


def test_lnl_pipeline(tmp_path):
    outdir = f"{tmp_path}/test_lnl_pipeline"
    os.makedirs(outdir, exist_ok=True)
    init_npts = 10
    _setup(asf_csv=PARAM_CSV, outdir=outdir, init_npts=init_npts)

    npts = 0
    while (npts < MAX_PTS):
        data = generate_training_data(asf_csv=PARAM_CSV, outdir=outdir)
        train_and_evaluate_model(train_data=data, outdir=outdir)
        npts = len(data)
