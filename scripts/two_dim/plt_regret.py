import numpy as np
import glob

import matplotlib.pyplot as plt
from lnl_computer.mock_data import load_mock_data

mock_data = load_mock_data("out_test")
print(mock_data.mcz_grid.n_bbh)
#
#
# files = glob.glob("out_*/y.txt")
# data = dict()
# for file in files:
#     # get label "out_{label}/y.txt"
#     label = file.split("/")[0]
#     label = label.split("_")[1]
#     data[label] = np.loadtxt(file)
#
# plt.figure()
# for label, obs in data.items():
#
#     accum = np.minimum.accumulate(np.abs(obs-mock_data.truth['lnl']))
#     plt.plot(accum, label=label)
# plt.xlabel("Iteration")
# plt.ylabel("Abs(LnL Error) Training set")
# plt.yscale("log")
# plt.legend()
# plt.savefig("lnl_error.png")