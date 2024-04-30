from .pp_test.main import PPTest
import click
from typing import List
from bilby.core.result import make_pp_plot
from tqdm.auto import tqdm
from bilby.core.result import Result
import glob


@click.command("setup_pp_test")
@click.option("-n", type=int, help="Number of injections")
@click.option("--outdir", type=str, help="Output directory")
@click.option("--compas_h5", type=str, help="Path to COMPAS h5 file")
@click.option("-p", "--params", type=str, help="List of parameters", multiple=True)
@click.option("--duration", type=float, help="Duration")
@click.option("--n_init", type=int, help="Number of initial points")
@click.option("--n_rounds", type=int, help="Number of rounds")
@click.option("--n_pts_per_round", type=int, help="Number of points per round")
@click.option("-a", "--acq_fns", type=str, help="List of acquisition functions", multiple=True)
@click.option("-t", "--time", type=str, help="Time to run the job", default="1:00:00")
@click.option("-m", "--mem", type=str, help="Memory to use", default="6G")
def setup_pp_test(
        n: int,
        outdir: str,
        compas_h5: str,
        params: List[str],
        duration: float,
        n_init: int,
        n_rounds: int,
        n_pts_per_round: int,
        acq_fns: List[str],
        time: str,
        mem: str
):
    PPTest(
        n=n,
        outdir=outdir,
        compas_h5=compas_h5,
        params=params,
        duration=duration,
        n_init=n_init,
        n_rounds=n_rounds,
        n_pts_per_round=n_pts_per_round,
        acq_fns=acq_fns,
        time=time,
        mem=mem
    )


@click.command("pp_test")
@click.argument(
    "results_regex",
    type=str,
)
@click.option(
    "-f",
    "--filename",
    type=str,
    help="Output filename",
)
def pp_test(results_regex, filename):
    results = glob.glob(results_regex)
    results = [Result.from_json(f) for f in tqdm(results)]
    npts_list = [r.meta_data['npts'] for r in results]
    # assert all the same
    assert all([n == npts[0] for n in npts]), f"All results must have the same number of points f{npts_list}"

    fig = make_pp_plot(
        results, save=False
    )
    fig.suptitle(f"N-Training pts: {npts_list[0]}")
    fig.savefig(filename)
