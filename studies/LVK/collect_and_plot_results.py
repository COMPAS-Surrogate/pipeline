import glob
import os
from bilby.core.result import read_in_result
import h5py
import pandas as pd
from collections import namedtuple
from pygtc import plotGTC

REGEX = "/fred/oz303/avajpeyi/studies/cosmic_int/lvk/out_*/out_mcmc/*pts_result.json"
VAR = "round{}_*pts_variable_lnl_result.json"

RES = namedtuple("RES", ["chains", "label"])

PARAM = ['aSF', 'dSF', 'sigma_0', 'mu_z']
PARAM_LATEX = [r"$a_{\rm SF}$", r"$d_{\rm SF}$", r"$\sigma_0$", r"$\mu_z$"]


def _get_vfname(fname, roundidx):
    vfname = fname.split("/")
    vfname[-1] = VAR.format(roundidx)
    vfname = "/".join(vfname)
    fnames = glob.glob(vfname)
    if len(fnames) == 0:
        raise ValueError(f"No variable lnl file found for {fname}")
    return fnames[0]


def collect_results(regex: str = REGEX):
    fnames = glob.glob(regex)
    r_group = []
    r_round = []
    var_fname = []
    for f in fnames:
        r_group.append(f.split("/")[-3].split("out_")[1])
        rid = int(f.split("/")[-1].split("_")[0].split("round")[1])
        r_round.append(rid)
        v = _get_vfname(f, rid)

        var_fname.append(v)
    data = pd.DataFrame(dict(
        fname=fnames,
        vfname=var_fname,
        group=r_group,
        round=r_round,
    ))
    data = data.loc[data.groupby("group")["round"].idxmax()]
    data = data.sort_values("group")
    data = data.reset_index(drop=True)

    data_sets = []
    # iterate over rows
    for i, row in data.iterrows():
        res = read_in_result(row.fname)
        vres = read_in_result(row.vfname)
        # make same size of posterior
        min_size = min(len(res.posterior), len(vres.posterior))
        # make same size of posterior
        chain1 = res.posterior.sample(min_size)[PARAM]
        chain2 = vres.posterior.sample(min_size)[PARAM]
        npts = res.meta_data["npts"]
        label = f"{row.group}_{npts}pts"
        fig = plotGTC(
            chains=[chain1, chain2],
            chainLabels=["gp_mu-LnL", "N(gp_mu, gp_sig)-LnL"],
            paramNames=PARAM_LATEX,
            truthLabels=label,
            plotName=f"{label}.pdf",
            figureSize='APJ_page'
        )



collect_results()
