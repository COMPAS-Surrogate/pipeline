from lnl_surrogate import LnLSurrogate
from pp_test.make_pp_plot import make_pp_plot, PP_DATA
import bilby

import glob
import os
import numpy
from tqdm.auto import tqdm
from lnl_surrogate.surrogate import build_surrogate
from pp_test.cli import pp_test
import sys


def get_csv_paths(regex):
    return glob.glob(regex)


def main(base_dir, regex, threshold_amt):
    csv = get_csv_paths(f"{base_dir}/{regex}")
    print(f"Generating thresholded-lnl posteriors for {len(csv)} csv files")
    if len(csv) == 0:
        raise ValueError(f"No CSV files found with the given regex: {base_dir}/{regex}")
    for c in tqdm(csv):
        c = os.path.abspath(c)
        outdir = os.path.join(os.path.dirname(c), "out_thresholded")
        os.makedirs(outdir, exist_ok=True)
        build_surrogate(csv=c, model_type="gp", outdir=outdir, lnl_threshold=threshold_amt,
                            label=f"thresh{threshold_amt}")
        print(f"Saved thresholded lnl surrogate model to {outdir}")
    pp_test(
        results_regex=f"{base_dir}/*/out_thresholded/*_result.json",
        cached_json=f"{base_dir}/thresholded_pp_plot.json",
        filename=f"{base_dir}/thresholded_pp_plot.png"
    )


if __name__ == "__main__":
    main(
        base_dir=sys.argv[1],
        regex=sys.argv[2],
        threshold_amt=float(sys.argv[3])
    )
