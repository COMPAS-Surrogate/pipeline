import bilby.core.result
from bilby.core.result import latex_plot_format, safe_save_figure
from itertools import product
from collections import namedtuple
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict
import json
import pickle

logger = logging.getLogger('bilby')

CONFIDENCE_INTERVAL = [0.68, 0.95, 0.997]
CI_ALPHA = [0.1] * len(CONFIDENCE_INTERVAL)
LEGEND_FONTSIZE = 'x-small'

PP_DATA = namedtuple(
    'pp_data',
    ['p_value', 'pp', 'label']
)
X_DATA = np.linspace(0, 1, 1001)


class PP_DATA:
    def __init__(self, p_value: float, pp: np.ndarray, label: str):
        self.p_value = p_value
        self.pp = pp
        self.label = label

    def to_dict(self):
        return dict(
            p_value=self.p_value,
            pp=self.pp,
            label=self.label
        )

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            p_value=data['p_value'],
            pp=data['pp'],
            label=data['label']
        )


def cache_pp_data(results: List[bilby.core.result.Result], filename: str) -> List[PP_DATA]:
    pp_data = []
    # Get CIs for each result
    keys = results[0].search_parameter_keys
    credible_levels = list()
    for i, result in enumerate(results):
        credible_levels.append(
            result.get_all_injection_credible_levels(keys)
        )
    credible_levels = pd.DataFrame(credible_levels)
    # Get pp for each CI
    for ii, key in enumerate(credible_levels):
        pp = np.array(
            [
                sum(credible_levels[key].values < xx) / len(credible_levels)
                for xx in X_DATA
            ])
        pvalue = scipy.stats.kstest(credible_levels[key], 'uniform').pvalue
        pp_data.append(PP_DATA(
            p_value=pvalue,
            pp=pp,
            label=f"{_get_label(key, results[0])} ({pvalue:2.3f})"
        ))

    save_pp_datalist(
        pp_datalist=pp_data,
        n_sim=len(results),
        n_gp=results[0].meta_data.get('npts', None),
        filename=filename
    )

    return pp_data


def save_pp_datalist(pp_datalist: List[PP_DATA], n_sim: int, n_gp: int, filename: str):
    data = dict(
        pp_data=[pp.to_dict() for pp in pp_datalist],
        n_sim=n_sim,
        n_gp=n_gp
    )
    with open(filename, 'wb') as f:
        # Use pickle.dump() to write the dictionary to the file
        pickle.dump(data, f)


def load_pp_datalist(filename: str):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    pp_data = [PP_DATA.from_dict(pp) for pp in data['pp_data']]
    n_gp = data.get('n_gp', None)
    n_sim = data.get('n_sim', None)
    return pp_data, n_sim, n_gp


@latex_plot_format
def make_pp_plot(
        pp_datafn: str,
        results: List[bilby.core.result.Result] = None,
        filename=None,
        lines=None,
        **kwargs):
    if results is not None:
        cache_pp_data(results, pp_datafn    )
    pp_data, n, n_gp = load_pp_datalist(pp_datafn)
    title = _get_title(combined_pval=_combined_pvalue(pp_data), n=n, surr_pts=n_gp)
    return _plot_pp(pp_data, n=n, lines=lines, title=title, filename=filename, **kwargs, )


def _combined_pvalue(pp_data):
    pvals = [pp.p_value for pp in pp_data]
    return scipy.stats.combine_pvalues(pvals)[1]


def _get_title(combined_pval, n, surr_pts=None):
    title = f"p-value={combined_pval:2.4f}, N sims={n}"
    if surr_pts:
        title += f", N GP-pts={surr_pts}"
    return title


def _get_linestyles():
    colors = [f"C{i}" for i in range(8)]
    linestyles = ["-", "--", ":"]
    lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    return lines


def _get_label(key, result):
    try:
        l = result.priors[key].latex_label
    except (AttributeError, KeyError):
        l = key
    return l


def _add_ci_bounds(ax, N):
    for ci, alpha in zip(CONFIDENCE_INTERVAL, CI_ALPHA):
        edge_of_bound = (1. - ci) / 2.
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, X_DATA) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, X_DATA) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(X_DATA, lower, upper, alpha=alpha, color='k')


def _plot_pp(pp_data: List[PP_DATA], n: int, lines=None, title=None, filename=None, **kwargs):
    if kwargs.get('fig', None) is not None:
        fig = kwargs.get('fig', None)
        ax = fig.gca()
    else:
        fig, ax = plt.subplots()
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    lines = _get_linestyles() if lines is None else lines

    _add_ci_bounds(ax, n)
    for i, pp_data_i in enumerate(pp_data):
        plt.plot(X_DATA, pp_data_i.pp, lines[i], label=pp_data_i.label, **kwargs)

    ax.set_title(title)
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.legend(handlelength=2, labelspacing=0.25, fontsize=LEGEND_FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if filename is not None:
        safe_save_figure(fig=fig, filename=filename, dpi=500)
    return fig
