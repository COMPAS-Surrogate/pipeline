import json
import os
from typing import Dict

from abc import ABC, abstractmethod
from .utils import _ensure_dir
from typing import List, Dict

import subprocess
import os


class BaseJob(ABC):

    def __init__(
            self,
            label: str,
            base_outdir: str,
            params: Dict[str, float],
            compas_h5: str,
            duration: float,
            **kwargs
    ):
        self.label = label
        self.base_outdir = base_outdir
        self.compas_h5 = compas_h5
        assert len(params) > 0, "params must be a non-empty dictionary"
        self.params_dict = params
        self.duration = duration



    @property
    def outdir(self) -> str:
        return _ensure_dir(os.path.join(self.base_outdir, f'out_surr_{self.label}'))

    @property
    def truth_json(self) -> str:
        fname = _ensure_dir(os.path.join(self.outdir, 'truth.json'))
        return fname

    @property
    def mock_obs_fname(self) -> str:
        return _ensure_dir(os.path.join(self.outdir, 'mock_obs.npz'))

    def pre_run_checks(self):
        assert os.path.exists(self.compas_h5), f"COMPAS file not found: {self.compas_h5}"
        assert os.path.exists(self.outdir), f"Output directory not found: {self.outdir}"

    def run(self):
        self.pre_run_checks()
        print(f"Running: {self.command}")
        subprocess.run(self.command, shell=True)
        print(f"Finished: {self.command}")

    @property
    @abstractmethod
    def params(self) -> str:
        pass

    @property
    @abstractmethod
    def job_kwargs(self) -> Dict[str, str]:
        pass

    @property
    @abstractmethod
    def command(self) -> str:
        pass


def _ensure_dir(path_str):
    if '.' in os.path.basename(path_str):  # if dirname has an extension, it is a filename
        dirname = os.path.dirname(path_str)
    else:
        dirname = path_str

    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    return path_str


def _write_json(d: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(d, f)
