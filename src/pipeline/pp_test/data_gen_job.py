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

MAKE_MOCK_OBS = 'make_mock_obs {compas_h5} --duration {duration} --fname {fname} --sf_sample "{sf_sample}" --save_plots'


class DataGenJob(BaseJob):
    @property
    def params(self) -> str:
        return ' '.join([f'{k}:{v}' for k, v in self.params_dict.items()])

    @property
    def job_kwargs(self) -> Dict[str, str]:
        """
        make_mock_obs
        {compas_h5}
        --sf_sample "{sf_sample}"
        --duration {duration}
        --fname {fname}
        """
        return dict(
            compas_h5=self.compas_h5,
            sf_sample=self.params,
            duration=self.duration,
            fname=self.mock_obs_fname
        )

    @property
    def command(self) -> str:
        return MAKE_MOCK_OBS.format(**self.job_kwargs)
