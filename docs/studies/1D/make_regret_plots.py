from lnl_surrogate.plotting.regret_plots import plot_multiple_regrets, RegretData
from lnl_computer.mock_data import load_mock_data

mock_data = load_mock_data('outdir')

labels = ['Explore', 'Exploit', 'Both']
params = ['dSF', 'mu_z', 'sigma_0', 'aSF']
colors = ['blue', 'orange', 'green']

data_fmt = 'outdir/gp_{param}/{label}/regret.csv'

for param in params:
    regret_data = [
        RegretData(data_fmt.format(param=param, label=labels[i].lower()), labels[i], colors[i]) for i in range(3)
    ]
    plot_multiple_regrets(
        regret_data, fname=f'outdir_rel/gp_{param}/regret.png',
        true_min=mock_data.reference_param['lnl'],
    )

