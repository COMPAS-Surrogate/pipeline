import os
import numpy as np
from lnl_computer.mock_data import MockData
from lnl_computer.cli.main import make_sf_table, batch_lnl_generation

np.random.seed(42)
OUTDIR = 'test_data'
PARAM_TABLE = f"{OUTDIR}/param_table.csv"

mock_data = MockData.generate_mock_datasets(OUTDIR)
mock_data.mcz_grid.plot().show()


make_sf_table(parameters=['aSF'], n=5, fname=PARAM_TABLE)
batch_lnl_generation(
    mcz_obs=mock_data.observations,
    compas_h5_path=mock_data.compas_filename,
    parameter_table=PARAM_TABLE,
    outdir=OUTDIR,
    n_bootstraps=0,
    save_images=False
)

import trieste
from trieste.objectives.utils import mk_observer

observer = mk_observer(function)

num_initial_points = 20
num_steps = 20
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)