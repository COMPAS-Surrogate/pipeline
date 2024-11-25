import argparse
from lnl_computer.cosmic_integration.mcz_grid import McZGrid


# JEFF_PARMS = dict(mu_z=-0.154, sigma_0=0.546, aSF=0.001, dSF=4.76)


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
