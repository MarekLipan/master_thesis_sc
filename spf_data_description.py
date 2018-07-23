"""
ECB Survey of Professional Forecasters

DATA DESCRIPTION SCRIPT

This script is used to create and export data descriptive charts and tables.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cycler import cycler
from scipy import stats
from pylatex import Table, Tabular
from pylatex.utils import NoEscape
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

# path to where the figures and tables are stored
fig_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Figures/"
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"

###############################
# DESCRIPTIVE STATISTICS TABLE#
###############################

ds_stats = ["Mean", "Median", "Mode", "Std. Dev.", "Variance",
            "Minimum", "Maximum", "Kurtosis", "Skewness"]
tickers = ['RGDP (1Y)', 'RGDP (2Y)', 'HICP (1Y)', 'HICP (2Y)',
           'UNEM (1Y)', 'UNEM (2Y)']

# table of descriptive statistics
spf_desc_stat = pd.DataFrame(data=np.full((9, 6), np.nan, dtype=np.float),
                             columns=tickers,
                             index=ds_stats)

spf_desc_stat.iloc[0, :] = np.array([
        np.mean(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        np.mean(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        np.mean(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        np.mean(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        np.mean(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        np.mean(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[1, :] = np.array([
        np.median(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        np.median(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        np.median(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        np.median(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        np.median(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        np.median(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[2, :] = np.array([
        stats.mode(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0)[0][0],
        stats.mode(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0)[0][0],
        stats.mode(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0)[0][0],
        stats.mode(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0)[0][0],
        stats.mode(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0)[0][0],
        stats.mode(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0)[0][0],
        ])
spf_desc_stat.iloc[3, :] = np.array([
        np.std(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        np.std(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        np.std(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        np.std(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        np.std(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        np.std(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[4, :] = np.array([
        np.var(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        np.var(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        np.var(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        np.var(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        np.var(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        np.var(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[5, :] = np.array([
        np.min(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        np.min(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        np.min(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        np.min(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        np.min(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        np.min(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[6, :] = np.array([
        np.max(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        np.max(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        np.max(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        np.max(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        np.max(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        np.max(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[7, :] = np.array([
        stats.kurtosis(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        stats.kurtosis(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        stats.kurtosis(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        stats.kurtosis(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        stats.kurtosis(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        stats.kurtosis(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])
spf_desc_stat.iloc[8, :] = np.array([
        stats.skew(spf_bal_RGDP_1Y.iloc[:, 0].values, axis=0),
        stats.skew(spf_bal_RGDP_2Y.iloc[:, 0].values, axis=0),
        stats.skew(spf_bal_HICP_1Y.iloc[:, 0].values, axis=0),
        stats.skew(spf_bal_HICP_2Y.iloc[:, 0].values, axis=0),
        stats.skew(spf_bal_UNEM_1Y.iloc[:, 0].values, axis=0),
        stats.skew(spf_bal_UNEM_2Y.iloc[:, 0].values, axis=0),
        ])

# create table object
tabl = Table()
tabl.add_caption("Descriptive statistics of the SPF target macroeconomic variables for the euro area")
tabl.append(NoEscape('\label{tab: spf_Desc_Stats}'))
# create tabular object
tabr = Tabular(table_spec="lcccccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row(["Statistic"] + tickers)
tabr.add_hline()
tabr.add_
# fill in the rows of tabular
for i in range(9):
    tabr.add_row([ds_stats[i]] + [
            "{:.1f}".format(item) for item in spf_desc_stat.iloc[i, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "spf_Desc_Stats")

###############
# END OF FILE #
###############
