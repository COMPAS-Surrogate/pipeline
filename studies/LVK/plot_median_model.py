import argparse
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation import load_observation
from lnl_computer.cosmic_integration.star_formation_paramters import DEFAULT_DICT
import pandas as pd
#
# FNAMES = {
#     "5M": "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_5M.h5",
#     "32M": "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_32M.h5",
#     "512M": "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_512M.h5",
# }
#
#
# data = []
# obs = load_observation("lvk")
# for key, fname in FNAMES.items():
#     print(f"Computing LNL for {key}")
#     lnl, unc = McZGrid.lnl(
#         sf_sample=DEFAULT_DICT,
#         mcz_obs=obs,
#         duration=1.0,
#         compas_h5_path=fname,
#         n_bootstraps=10,
#         outdir="outdir",
#     )
#     print(f"LNL for {key}: {lnl} +/- {unc}")
#     data.append({"size": key, "lnl": lnl, "unc": unc})
#
# df = pd.DataFrame(data)
# df.to_csv("lvk_lnl.csv")





# JEFF_PARMS = dict(mu_z=-0.154, sigma_0=0.546, aSF=0.001, dSF=4.76)
# Avi params = dict(mu_z=-0.16, sigma_0=0.56, aSF=0.01, dSF=4.70)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot median model")
    # get path to compas H5 file
    parser.add_argument("--compas-h5", type=str, help="Path to COMPAS H5 file", required=True)
    # get path to output fnme (give default name)
    parser.add_argument("--out", type=str, help="Path to output file", default="median_model.png")
    parser.add_argument("--aSF", type=float, help="aSF", default=0.001)
    parser.add_argument("--dSF", type=float, help="dSF", default=4.76)
    parser.add_argument("--mu_z", type=float, help="mu_z", default=-0.154)
    parser.add_argument("--sigma_0", type=float, help="sigma_0", default=0.546)
    return parser.parse_args()


def make_plot(compas_h5, out_fname, cosmic_kwargs):
    print(f"Making plot for {compas_h5} with {cosmic_kwargs} -> {out_fname}")

    mcz = McZGrid.from_compas_output(
        compas_path=compas_h5,
        cosmological_parameters=cosmic_kwargs,
    )
    fig0 = mcz.plot()
    fig0.savefig(out_fname.replace(".png", "_unbinned.png"))
    fig0.clear()
    mcz.bin_data()
    fig = mcz.plot()
    fig.savefig(out_fname)


def main():
    args = parse_args()
    cosmic_kwargs = {
        "aSF": args.aSF,
        "dSF": args.dSF,
        "mu_z": args.mu_z,
        "sigma_0": args.sigma_0,
    }
    make_plot(args.compas_h5, args.out, cosmic_kwargs)


if __name__ == '__main__':
    main()
