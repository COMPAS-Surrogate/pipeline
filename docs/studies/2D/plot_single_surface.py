import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lnl_computer.cosmic_integration.star_formation_paramters import STAR_FORMATION_RANGES
from lnl_surrogate import LnLSurrogate

out = "outdir/gp_mu_z_sigma_0"
lnl_surrogate = LnLSurrogate.load(out)

x = np.linspace(*STAR_FORMATION_RANGES['mu_z'], 100)
y = np.linspace(*STAR_FORMATION_RANGES['sigma_0'], 100)
xx, yy = np.meshgrid(x, y)
samples = np.c_[xx.flatten(), yy.flatten()]
samples = samples.astype(np.float64)

model_out, model_unc = lnl_surrogate.model.predict(samples)
model_out = model_out.numpy().flatten()

fig = go.Figure(data=[go.Surface(z=model_out.reshape(100, 100), x=x, y=y)])
fig.show()