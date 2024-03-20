import os
from typing import Tuple

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import trieste
from bilby.core.likelihood import Likelihood
from tqdm.auto import trange
from trieste.acquisition import EfficientGlobalOptimization, PredictiveVariance, \
    ExpectedImprovement
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.objectives import mk_observer
from trieste.observer import Observer
from trieste.space import SearchSpace
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

import numpy as np
from lnl_surrogate.surrogate import train, load

MIN_LIKELIHOOD_VARIANCE = 1e-6


class SurrogateLikelihood(Likelihood):
    def __init__(self, lnl_surrogate: "Model", parameter_keys: list, lnl_at_true=0):
        super().__init__({k: 0 for k in parameter_keys})
        self.param_keys = parameter_keys
        self.surr = lnl_surrogate
        self.lnl_at_true = lnl_at_true

    def log_likelihood(self):
        params = np.array([[self.parameters[k] for k in self.param_keys]])
        y_mean, y_std = self.surr.predict(params)
        y_mean = y_mean.numpy().flatten()[0]
        # this is the relative negative log likelihood, so we need to multiply by -1 and add the true likelihood
        return y_mean * -1 + self.lnl_at_true


surr = load('out_both/')
surrogate_likelihood = SurrogateLikelihood(surr, PARAMS)

result = bilby.run_sampler(
    likelihood=surrogate_likelihood,
    priors=get_star_formation_prior(),
    sampler="dynesty",
    npoints=250,
    injection_parameters=TRUE,
    outdir='out_test/out_sampler',
    label='surrogate',
    clean=True
)
result.plot_corner(truth=True)