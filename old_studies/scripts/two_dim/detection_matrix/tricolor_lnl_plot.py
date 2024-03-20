import pandas as pd
import glob
import matplotlib.pyplot as plt


def read_lnl_data()->pd.DataFrame:
    files = glob.glob("*.csv")
    # read all files as CSVs and append
    dataframes = [pd.read_csv(f) for f in files]
    df =  pd.concat(dataframes)
    df = df[df['sigma_0'] > 0.2]
    return df



if __name__ == '__main__':
    df = read_lnl_data()
    plt.tripcolor(df.aSF, df.sigma_0, df.lnl, shading='gouraud')
    plt.show()