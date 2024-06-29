from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation import load_observation
from lnl_computer.cosmic_integration.star_formation_paramters import DEFAULT_DICT
import pandas as pd
import

FNAMES = {
    "5M": "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_5M.h5",
    "32M": "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_32M.h5",
    "512M": "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_512M.h5",
}


data = []
obs = load_observation("lvk")
for key, fname in FNAMES.items():
    print(f"Computing LNL for {key}")
    lnl, unc = McZGrid.lnl(
        sf_sample=DEFAULT_DICT,
        mcz_obs=obs,
        duration=1.0,
        compas_h5_path=fname,
        n_bootstraps=10,
        outdir="outdir",
    )
    print(f"LNL for {key}: {lnl} +/- {unc}")
    data.append({"size": key, "lnl": lnl, "unc": unc})

df = pd.DataFrame(data)
df.to_csv("lvk_lnl.csv")

