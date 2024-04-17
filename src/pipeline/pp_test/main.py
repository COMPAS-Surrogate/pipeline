"""Run PP Test for the COMPAS LnL-Pipeline

1. Generate the "injection" file (i.e. the file that contains the true parameters) + mock observations for each
2. Run the COMPAS-LnL surrogate builder using data from the "injection" files
3. Sample the COMPAS-LnL surrogate models
4. Make PP-Plots





## The above produces the bilby results we'll need for PP-tests... this can be done manually for now


"""

import json
import pandas as pd
import numpy as np
import os
import shutil
from typing import List, Dict
import os
from .utils import _ensure_dir
from lnl_computer.cli.main import make_sf_table
import numpy as np

from .data_analysis_job import AnalysisJob
from .data_gen_job import DataGenJob

DIR = os.path.dirname(os.path.abspath(__file__))
SLURM_TEMPLATE = os.path.join(DIR, 'templates/slurm_template.sh')
BASH_TEMPLATE = os.path.join(DIR, 'templates/bash_template.sh')

np.random.seed(0)


class PPTest:

    def __init__(
            self,
            n: int,
            outdir: str,
            compas_h5: str,
            params: List[str],
            duration: float,
            n_init: int,
            n_rounds: int, n_pts_per_round: int, acq_fns: List[str],
    ):
        self.n = n
        self.outdir = _ensure_dir(outdir)
        self.compas_h5 = compas_h5
        self.params = params
        self.injection_params = self._load_injections()

        self.job_kwgs = dict(
            base_outdir=self.outdir,
            compas_h5=self.compas_h5,
            duration=duration,
            n_init=n_init,
            n_rounds=n_rounds,
            n_pts_per_round=n_pts_per_round,
            acq_fns=acq_fns
        )

        self._write_jobs()

    @property
    def injection_file(self):
        fname = os.path.join(self.outdir, 'injection.csv')
        if not os.path.exists(fname):
            make_sf_table(
                parameters=self.params,
                n=self.n,
                grid_parameterspace=False,
                fname=os.path.join(self.outdir, 'injection.csv')
            )
        return fname

    @property
    def data_gen_cmd_file(self):
        return os.path.join(self.outdir, 'gen_cmd.txt')

    @property
    def analysis_cmd_file(self):
        return os.path.join(self.outdir, 'analy_cmd.txt')

    def _load_injections(self) -> List[Dict[str, float]]:
        return pd.read_csv(self.injection_file).to_dict(orient='records')

    def _get_data_gen_cmds(self):
        return [
            DataGenJob(label=str(i), params=p, **self.job_kwgs).command
            for i, p in enumerate(self.injection_params)
        ]

    def _get_analysis_cmds(self):
        return [
            AnalysisJob(label=str(i), params=p, **self.job_kwgs).command
            for i, p in enumerate(self.injection_params)
        ]

    def _write_jobs(self):
        # write data-gen and analysis txt file
        with open(self.data_gen_cmd_file, 'w') as f:
            f.write('\n'.join(self._get_data_gen_cmds()))
        with open(self.analysis_cmd_file, 'w') as f:
            f.write('\n'.join(self._get_analysis_cmds()))

        self._write_bash()
        self._write_slurm()

        print(f"Run the following command to start the PP-Test:\n")
        print(f"sbatch {os.path.join(self.outdir, 'slurm_submit.sh')}")

    def _write_bash(self):
        # read the bash template
        with open(BASH_TEMPLATE, 'r') as f:
            template = f.read()

        template = template.replace('{{GEN_CMD_FILE}}', self.data_gen_cmd_file)
        template = template.replace('{{ANALY_CMD_FILE}}', self.analysis_cmd_file)
        with open(os.path.join(self.outdir, 'bash_run.sh'), 'w') as f:
            f.write(template)


    def _write_slurm(self):
        # read the slurm template
        with open(SLURM_TEMPLATE, 'r') as f:
            template = f.read()
        # replace the {{LOG}} and {{NJOBS}}
        log_dir = _ensure_dir(os.path.join(self.outdir, 'logs'))
        template = template.replace('{{LOG}}', log_dir)
        template = template.replace('{{NJOBS}}', str(self.n))
        template = template.replace('{{GEN_CMD_FILE}}', self.data_gen_cmd_file)
        template = template.replace('{{ANALY_CMD_FILE}}', self.analysis_cmd_file)
        # write the slurm file
        with open(os.path.join(self.outdir, 'slurm_submit.sh'), 'w') as f:
            f.write(template)
