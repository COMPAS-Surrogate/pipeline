import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot median model")
    # get path to compas H5 file
    parser.add_argument("--compas-h5", type=str, help="Path to COMPAS H5 file", required=True)
    # get path to output fnme (give default name)
    parser.add_argument("--out", type=str, help="Path to output file", default="median_model.png")
    return parser.parse_args()


def make_plot(compas_h5, out_fname):
    from lnl_computer.cosmic_integration.mcz_grid import McZGrid

    mcz = McZGrid.from_compas_output(
        compas_path=compas_h5,
        cosmological_parameters=dict(aSF=0.014, dSF=5, mu_z=-0.2, sigma_z=0.6),
    )
    fig0 = mcz.plot()
    fig0.savefig(out_fname.replace(".png", "_unbinned.png"))
    fig0.clear()
    mcz.bin_data()
    fig = mcz.plot()
    fig.savefig(out_fname)


def main():
    args = parse_args()
    make_plot(args.compas_h5, args.out)


if __name__ == '__main__':
    main()
