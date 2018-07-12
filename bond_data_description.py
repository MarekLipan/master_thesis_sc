"""
US Bond Volatility Forecasting

DATA DESCRIPTION SCRIPT

This script is used to create and export data descriptive charts and tables.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pylatex import Table, Tabular

# path to where the figures and tables are stored
fig_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Figures/"
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"

################
# SERIES CHARTS#
################

# RETURNS
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
# TU -  2 year
ret['TU'].plot(ax=axes[0, 0], color="black", linewidth=0.5)
a = axes[0, 0]
a.set_title('TU (2 Year)')
a.set_ylabel('Returns')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# FV -  5 year
ret['FV'].plot(ax=axes[0, 1], color="black", linewidth=0.5)
a = axes[0, 1]
a.set_title('FV (5 Year)')
a.set_ylabel('Returns')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# TY - 10 year
ret['TY'].plot(ax=axes[1, 0], color="black", linewidth=0.5)
a = axes[1, 0]
a.set_title('TY (10 Year)')
a.set_ylabel('Returns')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# US - 30 year
ret['US'].plot(ax=axes[1, 1], color="black", linewidth=0.5)
a = axes[1, 1]
a.set_title('US (30 Year)')
a.set_ylabel('Returns')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# whole figure
fig.autofmt_xdate()
fig.savefig(fig_path + "bond_ret_series.pdf", bbox_inches='tight')


# REALIZED VOLATILITY
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
# TU -  2 year
rvol['TU'].plot(ax=axes[0, 0], color="black", linewidth=0.5)
a = axes[0, 0]
a.set_title('TU (2 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# FV -  5 year
rvol['FV'].plot(ax=axes[0, 1], color="black", linewidth=0.5)
a = axes[0, 1]
a.set_title('FV (5 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# TY - 10 year
rvol['TY'].plot(ax=axes[1, 0], color="black", linewidth=0.5)
a = axes[1, 0]
a.set_title('TY (10 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# US - 30 year
rvol['US'].plot(ax=axes[1, 1], color="black", linewidth=0.5)
a = axes[1, 1]
a.set_title('US (30 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
# whole figure
fig.autofmt_xdate()
fig.savefig(fig_path + "bond_rvol_series.pdf", bbox_inches='tight')

###############################
# DESCRIPTIVE STATISTICS TABLE#
###############################

ds_stats = ["Mean", "Median", "Mode", "Standard Deviation", "Variance",
            "Minimum", "Maximum", "Kurtosis", "Skewness"]
tickers = list(ret)


# RETURNS
ret_desc_stat = pd.DataFrame(data=np.full((9, 4), np.nan, dtype=np.float),
                             columns=tickers,
                             index=ds_stats)
ret_desc_stat.iloc[0, :] = np.mean(ret.values, axis=0)
ret_desc_stat.iloc[1, :] = np.median(ret.values, axis=0)
ret_desc_stat.iloc[2, :] = stats.mode(ret.values, axis=0)[0]
ret_desc_stat.iloc[3, :] = np.std(ret.values, axis=0)
ret_desc_stat.iloc[4, :] = np.var(ret.values, axis=0)
ret_desc_stat.iloc[5, :] = np.min(ret.values, axis=0)
ret_desc_stat.iloc[6, :] = np.max(ret.values, axis=0)
ret_desc_stat.iloc[7, :] = stats.kurtosis(ret.values, axis=0)
ret_desc_stat.iloc[8, :] = stats.skew(ret.values, axis=0)

# create tabule object
tabl = Table()
tabl.add_caption("Returns -- Descriptive Statistics")
# create tabular object
tabr = Tabular(table_spec="lcccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row(["Statistic"] + tickers)
tabr.add_hline()
# fill in the rows of tabular
for i in range(9):
    tabr.add_row([ds_stats[i]] + [
            "{:.7f}".format(item) for item in ret_desc_stat.iloc[i, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_ret_desc_stats")

# REALIZED VOLATILITY
rvol_desc_stat = pd.DataFrame(data=np.full((9, 4), np.nan, dtype=np.float),
                             columns=tickers,
                             index=ds_stats)
rvol_desc_stat.iloc[0, :] = np.mean(rvol.values, axis=0)
rvol_desc_stat.iloc[1, :] = np.median(rvol.values, axis=0)
rvol_desc_stat.iloc[2, :] = stats.mode(rvol.values, axis=0)[0]
rvol_desc_stat.iloc[3, :] = np.std(rvol.values, axis=0)
rvol_desc_stat.iloc[4, :] = np.var(rvol.values, axis=0)
rvol_desc_stat.iloc[5, :] = np.min(rvol.values, axis=0)
rvol_desc_stat.iloc[6, :] = np.max(rvol.values, axis=0)
rvol_desc_stat.iloc[7, :] = stats.kurtosis(rvol.values, axis=0)
rvol_desc_stat.iloc[8, :] = stats.skew(rvol.values, axis=0)

# create tabule object
tabl = Table()
tabl.add_caption("Realized Volatility -- Descriptive Statistics")
# create tabular object
tabr = Tabular(table_spec="lcccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row(["Statistic"] + tickers)
tabr.add_hline()
# fill in the rows of tabular
for i in range(9):
    tabr.add_row([ds_stats[i]] + [
            "{:.7f}".format(item) for item in rvol_desc_stat.iloc[i, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_rvol_desc_stats")

###############
# END OF FILE #
###############