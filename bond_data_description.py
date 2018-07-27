"""
US Bond Volatility Forecasting

DATA DESCRIPTION SCRIPT

This script is used to create and export data descriptive charts and tables.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import accuracy_measures as am
from cycler import cycler
from scipy import stats
from pylatex import Table, Tabular, MultiColumn, MultiRow
from pylatex.utils import NoEscape
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

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

ds_stats = ["Mean", "Median", "Mode", "Std. Dev.", "Variance",
            "Minimum", "Maximum", "Kurtosis", "Skewness"]
tickers = ['TU (2 Year)', 'FV (5 Year)', 'TY (10 Year)', 'US (30 Year)']


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

# create table object
tabl = Table()
tabl.add_caption("Descriptive statistics of log-returns of U.S. Treasury futures")
tabl.append(NoEscape('\label{tab: Bond_ret_Desc_Stats}'))
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

# create table object
tabl = Table()
tabl.add_caption("Descriptive statistics of realized volatility of log-returns of U.S. Treasury futures")
tabl.append(NoEscape('\label{tab: Bond_RVOL_Desc_Stats}'))
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

###########
# ACF/PACF#
###########

plot_acf(rvol.iloc[:,3], lags=20)
plot_pacf(rvol.iloc[:,3], lags=20)

############
# ADF TESTS#
############
adf_stats = ["Test Statistic", "P-value"]

# RETURNS
adf_ret = pd.DataFrame(data=np.full((2, 4), np.nan, dtype=float),
                       columns=tickers,
                       index=adf_stats)

for i in range(4):
    adf_test = adfuller(ret.iloc[:, i])
    adf_ret.iloc[0, i] = adf_test[0]
    adf_ret.iloc[1, i] = adf_test[1]

# create table object
tabl = Table()
tabl.add_caption("Augmented Dickey-Fuller test results for the log-returns of U.S. Treasury futures")
tabl.append(NoEscape('\label{tab: Bond_ret_ADF}'))
# create tabular object
tabr = Tabular(table_spec="lcccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row([""] + tickers)
tabr.add_hline()
# fill in the rows of tabular
for i in range(2):
    tabr.add_row([adf_stats[i]] + [
            "{:.4f}".format(item) for item in adf_ret.iloc[i, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "Bond_ret_ADF")

# REALIZED VOLATILITY
adf_rvol = pd.DataFrame(data=np.full((2, 4), np.nan, dtype=float),
                        columns=tickers,
                        index=adf_stats)

for i in range(4):
    adf_test = adfuller(rvol.iloc[:, i])
    adf_rvol.iloc[0, i] = adf_test[0]
    adf_rvol.iloc[1, i] = adf_test[1]

# create table object
tabl = Table()
tabl.add_caption("Augmented Dickey-Fuller test results for the realized volatility of log-returns of U.S. Treasury futures")
tabl.append(NoEscape('\label{tab: Bond_RVOL_ADF}'))
# create tabular object
tabr = Tabular(table_spec="lcccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row([""] + tickers)
tabr.add_hline()
# fill in the rows of tabular
for i in range(2):
    tabr.add_row([adf_stats[i]] + [
            "{:.4f}".format(item) for item in adf_rvol.iloc[i, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "Bond_RVOL_ADF")

#######################
# INDIVIDUAL FORECASTS#
#######################
# number of individual forecasts
K = ind_fcts_1_TU.shape[1]

# PLOT OF INDIVIDUAL FORECASTS
# colour cycle
cyc = cycler('color', ['red', 'cyan', 'magenta', 'yellow','black',
                       'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'darkgrey',
                       'dimgrey', 'dimgrey', 'dimgrey'])
# 1-step-ahead forecasts
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
# TU -  2 year
a = axes[0, 0]
a.set_prop_cycle(cyc)
ind_fcts_1_TU.plot(ax=a, linewidth=0.5)
a.set_title('TU (2 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# FV -  5 year
a = axes[0, 1]
a.set_prop_cycle(cyc)
ind_fcts_1_FV.plot(ax=a, linewidth=0.5)
a.set_title('FV (5 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# TY - 10 year
a = axes[1, 0]
a.set_prop_cycle(cyc)
ind_fcts_1_TY.plot(ax=a, linewidth=0.5)
a.set_title('TY (10 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# US - 30 year
a = axes[1, 1]
a.set_prop_cycle(cyc)
ind_fcts_1_US.plot(ax=a, linewidth=0.5)
a.set_title('US (30 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# whole figure
fig.autofmt_xdate()
fig.savefig(fig_path + "bond_ind_fcts_1.pdf", bbox_inches='tight')

# 5-step-ahead forecasts
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
# TU -  2 year
a = axes[0, 0]
a.set_prop_cycle(cyc)
ind_fcts_5_TU.plot(ax=a, linewidth=0.5)
a.set_title('TU (2 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# FV -  5 year
a = axes[0, 1]
a.set_prop_cycle(cyc)
ind_fcts_5_FV.plot(ax=a, linewidth=0.5)
a.set_title('FV (5 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# TY - 10 year
a = axes[1, 0]
a.set_prop_cycle(cyc)
ind_fcts_5_TY.plot(ax=a, linewidth=0.5)
a.set_title('TY (10 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# US - 30 year
a = axes[1, 1]
a.set_prop_cycle(cyc)
ind_fcts_5_US.plot(ax=a, linewidth=0.5)
a.set_title('US (30 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# whole figure
fig.autofmt_xdate()
fig.savefig(fig_path + "bond_ind_fcts_5.pdf", bbox_inches='tight')

# 22-step-ahead forecasts
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
# TU -  2 year
a = axes[0, 0]
a.set_prop_cycle(cyc)
ind_fcts_22_TU.plot(ax=a, linewidth=0.5)
a.set_title('TU (2 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# FV -  5 year
a = axes[0, 1]
a.set_prop_cycle(cyc)
ind_fcts_22_FV.plot(ax=a, linewidth=0.5)
a.set_title('FV (5 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# TY - 10 year
a = axes[1, 0]
a.set_prop_cycle(cyc)
ind_fcts_22_TY.plot(ax=a, linewidth=0.5)
a.set_title('TY (10 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# US - 30 year
a = axes[1, 1]
a.set_prop_cycle(cyc)
ind_fcts_22_US.plot(ax=a, linewidth=0.5)
a.set_title('US (30 Year)')
a.set_ylabel('RVOL')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc=1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)
# whole figure
fig.autofmt_xdate()
fig.savefig(fig_path + "bond_ind_fcts_22.pdf", bbox_inches='tight')

# TABLE OF INDIVIDUAL FORECAST PERFORMANCE
# prepare individual performance tables
# TU
ind_TU_perf = pd.DataFrame(data=np.full((12, 9), 0, dtype=float),
                             columns=["RMSE", "MAE", "MAPE"]*3,
                             index=list(ind_fcts_1_TU)[1:])
for i in range(1, 13):
    # w = 1
    real_values = ind_fcts_1_TU.iloc[:, 0].values
    errors = real_values - ind_fcts_1_TU.iloc[:, i].values
    ind_TU_perf.iloc[i-1, :3] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 5
    real_values = ind_fcts_5_TU.iloc[:, 0].values
    errors = real_values - ind_fcts_5_TU.iloc[:, i].values
    ind_TU_perf.iloc[i-1, 3:6] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 22
    real_values = ind_fcts_22_TU.iloc[:, 0].values
    errors = real_values - ind_fcts_22_TU.iloc[:, i].values
    ind_TU_perf.iloc[i-1, 6:9] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
# FV
ind_FV_perf = pd.DataFrame(data=np.full((12, 9), 0, dtype=float),
                             columns=["RMSE", "MAE", "MAPE"]*3,
                             index=list(ind_fcts_1_FV)[1:])
for i in range(1, 13):
    # w = 1
    real_values = ind_fcts_1_FV.iloc[:, 0].values
    errors = real_values - ind_fcts_1_FV.iloc[:, i].values
    ind_FV_perf.iloc[i-1, :3] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 5
    real_values = ind_fcts_5_FV.iloc[:, 0].values
    errors = real_values - ind_fcts_5_FV.iloc[:, i].values
    ind_FV_perf.iloc[i-1, 3:6] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 22
    real_values = ind_fcts_22_FV.iloc[:, 0].values
    errors = real_values - ind_fcts_22_FV.iloc[:, i].values
    ind_FV_perf.iloc[i-1, 6:9] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
# TY
ind_TY_perf = pd.DataFrame(data=np.full((12, 9), 0, dtype=float),
                             columns=["RMSE", "MAE", "MAPE"]*3,
                             index=list(ind_fcts_1_TY)[1:])
for i in range(1, 13):
    # w = 1
    real_values = ind_fcts_1_TY.iloc[:, 0].values
    errors = real_values - ind_fcts_1_TY.iloc[:, i].values
    ind_TY_perf.iloc[i-1, :3] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 5
    real_values = ind_fcts_5_TY.iloc[:, 0].values
    errors = real_values - ind_fcts_5_TY.iloc[:, i].values
    ind_TY_perf.iloc[i-1, 3:6] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 22
    real_values = ind_fcts_22_TY.iloc[:, 0].values
    errors = real_values - ind_fcts_22_TY.iloc[:, i].values
    ind_TY_perf.iloc[i-1, 6:9] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
# US
ind_US_perf = pd.DataFrame(data=np.full((12, 9), 0, dtype=float),
                             columns=["RMSE", "MAE", "MAPE"]*3,
                             index=list(ind_fcts_1_US)[1:])
for i in range(1, 13):
    # w = 1
    real_values = ind_fcts_1_US.iloc[:, 0].values
    errors = real_values - ind_fcts_1_US.iloc[:, i].values
    ind_US_perf.iloc[i-1, :3] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 5
    real_values = ind_fcts_5_US.iloc[:, 0].values
    errors = real_values - ind_fcts_5_US.iloc[:, i].values
    ind_US_perf.iloc[i-1, 3:6] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])
    # w = 22
    real_values = ind_fcts_22_US.iloc[:, 0].values
    errors = real_values - ind_fcts_22_US.iloc[:, i].values
    ind_US_perf.iloc[i-1, 6:9] = np.array([
                am.RMSE(errors)*10000,
                am.MAE(errors)*10000,
                am.MAPE(errors, real_values)
                ])

# print the table
# create table object
tabl = Table()
tabl.add_caption("Forecast performance (measured in terms of RMSE, MAE and MAPE) of indindividual volatility models in h-steps-ahead forecasting of the realized volatility of U.S. Treasury futures log-returns")
tabl.append(NoEscape('\label{tab: bond_ind_perf}'))
# create tabular object
tabr = Tabular(table_spec="c|l|ccc|ccc|ccc")
tabr.add_hline()
tabr.add_hline()
# header row

tabr.add_row((MultiRow(2, data="Future"), MultiRow(2, data="Volatility Model"),
              MultiColumn(3, align='c|', data="h = 1"),
              MultiColumn(3, align='c|', data="h = 5"),
              MultiColumn(3, align='c', data="h = 22")))
tabr.add_hline(start=3, end=11, cmidruleoption="lr")
tabr.add_row([""]*2 + ["RMSE", "MAE", "MAPE"]*3)
tabr.add_hline()
# fill in the rows of tabular
tabr.add_row([MultiRow(12, data="TU (2 Year)")] + [ind_TU_perf.index[0]] + [
            "{:.2f}".format(item) for item in ind_TU_perf.iloc[0, :]])
for i in range(1, 12):
    tabr.add_row([""] + [ind_TU_perf.index[i]] + [
            "{:.2f}".format(item) for item in ind_TU_perf.iloc[i, :]])

tabr.add_hline()
tabr.add_row([MultiRow(12, data="FV (5 Year)")] + [ind_FV_perf.index[0]] + [
            "{:.2f}".format(item) for item in ind_FV_perf.iloc[0, :]])
for i in range(1, 12):
    tabr.add_row([""] + [ind_FV_perf.index[i]] + [
            "{:.2f}".format(item) for item in ind_FV_perf.iloc[i, :]])

tabr.add_hline()
tabr.add_row([MultiRow(12, data="TY (10 Year)")] + [ind_TY_perf.index[0]] + [
            "{:.2f}".format(item) for item in ind_TY_perf.iloc[0, :]])
for i in range(1, 12):
    tabr.add_row([""] + [ind_TY_perf.index[i]] + [
            "{:.2f}".format(item) for item in ind_TY_perf.iloc[i, :]])

tabr.add_hline()
tabr.add_row([MultiRow(12, data="US (30 Year)")] + [ind_US_perf.index[0]] + [
            "{:.2f}".format(item) for item in ind_US_perf.iloc[0, :]])
for i in range(1, 12):
    tabr.add_row([""] + [ind_US_perf.index[i]] + [
            "{:.2f}".format(item) for item in ind_US_perf.iloc[i, :]])

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_ind_perf")

###############
# END OF FILE #
###############
