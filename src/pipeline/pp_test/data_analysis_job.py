"""Run PP Test for the COMPAS LnL-Pipeline

1. Generate the "injection" file (i.e. the file that contains the true parameters) + mock observations for each
2. Run the COMPAS-LnL surrogate builder using data from the "injection" files
3. Sample the COMPAS-LnL surrogate models
4. Make PP-Plots





## The above produces the bilby results we'll need for PP-tests... this can be done manually for now


"""

from abc import ABC, abstractmethod
from .utils import _ensure_dir, BaseJob
from typing import List, Dict

import subprocess
import os



LNL_SURROGATE_RUNNER = 'train_lnl_surrogate --compas_h5_filename {compas_h5} --mcz_obs {mcz_obs} -o {outdir} --n_init {n_init} --n_rounds {n_rounds} --n_pts_per_round {n_pts_per_round} --save_plots --truth {truth} --duration {duration} {acquisition_fns} {params}'

class AnalysisJob(BaseJob):

    def __init__(self,
                 n_init: int,
                 n_rounds: int, n_pts_per_round: int, acq_fns: List[str],
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.n_init = n_init
        self.n_rounds = n_rounds
        self.n_pts_per_round = n_pts_per_round

        assert n_init > 1, "n_init must be greater than 1"
        assert n_rounds > 0, "n_rounds must be greater than 0"
        assert n_pts_per_round > 0, "n_pts_per_round must be greater than 0"

        assert len(acq_fns) > 0, "acq_fns must be a non-empty list"

        # acq fns must be a list of strings, joined like "-a pv -a ei"
        self.acq_fns = ' '.join([f'-a {fn}' for fn in acq_fns])

    @property
    def params(self) -> str:
        return ' '.join([f'-p {k}' for k, v in self.params_dict.items()])

    @property
    def job_kwargs(self) -> Dict[str, str]:
        """
        train_lnl_surrogate
        --compas_h5_filename {compas_h5}
        --mcz_obs {mcz_obs}
        -o {outdir}
        --duration {duration}
        --n_init {n_init}
        --n_rounds {n_rounds}
        --n_pts_per_round {n_pts_per_round}
        --save_plots
        --truth {truth} {acquisition_fns} {params}'
        """
        return dict(
            compas_h5=self.compas_h5,
            mcz_obs=self.mock_obs_fname,
            outdir=self.outdir,
            duration=self.duration,
            n_init=self.n_init,
            n_rounds=self.n_rounds,
            n_pts_per_round=self.n_pts_per_round,
            truth=self.truth_json,
            acquisition_fns=self.acq_fns,
            params=self.params
        )

    @property
    def command(self) -> str:
        return LNL_SURROGATE_RUNNER.format(**self.job_kwargs)

    def pre_run_checks(self):
        super().pre_run_checks()
        assert os.path.exists(self.mock_obs_fname), f"Mock Observations file not found: {self.mock_obs_fname}"
        assert os.path.exists(self.truth_json), f"Truth file not found: {self.truth_json}"
