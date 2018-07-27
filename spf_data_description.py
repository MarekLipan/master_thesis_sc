"""
ECB Survey of Professional Forecasters

DATA DESCRIPTION SCRIPT

This script is used to create and export data descriptive charts and tables.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import accuracy_measures as am
from cycler import cycler
from scipy import stats
from pylatex import Table, Tabular, MultiColumn, MultiRow, Tabu
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
tabr = Tabular(table_spec="l|cc|cc|cc")
tabr.add_hline()
tabr.add_hline()
# header row

tabr.add_row((MultiRow(2, data="Statistic"), MultiColumn(2, align='|c|', data="RGDP"),
              MultiColumn(2, align='|c|', data="HICP"),
              MultiColumn(2, align='|c', data="UNEM")))
tabr.add_hline(start=2, end=3, cmidruleoption="lr")
tabr.add_hline(start=4, end=5, cmidruleoption="lr")
tabr.add_hline(start=6, end=7, cmidruleoption="lr")
tabr.add_row([""] + ["1Y", "2Y"] + ["1Y", "2Y"] + ["1Y", "2Y"])
tabr.add_hline()
# fill in the rows of tabular
for i in range(9):
    tabr.add_row([ds_stats[i]] + [
            "{:.2f}".format(item) for item in spf_desc_stat.iloc[i, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "spf_Desc_Stats")


#########################################
# PLOT OF INDIVIDUAL FORECASTS VS TARGET#
#########################################

# SPF individual forecasts and the target
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
# RGDP (1Y)
a = axes[0, 0]
col_cyc = cycler('color', ['red']+['dimgrey']*spf_bal_RGDP_1Y.shape[1])
mar_cyc = cycler('marker', ['.']+['']*spf_bal_RGDP_1Y.shape[1])
a.set_prop_cycle(col_cyc+mar_cyc)
spf_bal_RGDP_1Y.plot(ax=a, linewidth=0.5)
a.set_title('Real GDP Growth (1 Year Horizon)')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend([Line2D([0], [0], color='red', lw=1, marker='.'),
          Line2D([0], [0], color='dimgrey', lw=1)],
         ['RGDP', 'Individual Forecasts'],
         loc=3, ncol=1, prop={'size': 8}, framealpha=1, fancybox=True)
# RGDP (2Y)
a = axes[0, 1]
col_cyc = cycler('color', ['red']+['dimgrey']*spf_bal_RGDP_2Y.shape[1])
mar_cyc = cycler('marker', ['.']+['']*spf_bal_RGDP_2Y.shape[1])
a.set_prop_cycle(col_cyc+mar_cyc)
spf_bal_RGDP_2Y.plot(ax=a, linewidth=0.5)
a.set_title('Real GDP Growth (2 Year Horizon)')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend([Line2D([0], [0], color='red', lw=1, marker='.'),
          Line2D([0], [0], color='dimgrey', lw=1)],
         ['RGDP', 'Individual Forecasts'],
         loc=3, ncol=1, prop={'size': 8}, framealpha=1, fancybox=True)
# HICP (1Y)
a = axes[1, 0]
col_cyc = cycler('color', ['red']+['dimgrey']*spf_bal_HICP_1Y.shape[1])
mar_cyc = cycler('marker', ['.']+['']*spf_bal_HICP_1Y.shape[1])
a.set_prop_cycle(col_cyc+mar_cyc)
spf_bal_HICP_1Y.plot(ax=a, linewidth=0.5)
a.set_title('Harmonised Inflation (1 Year Horizon)')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend([Line2D([0], [0], color='red', lw=1, marker='.'),
          Line2D([0], [0], color='dimgrey', lw=1)],
         ['HICP', 'Individual Forecasts'],
         loc=3, ncol=1, prop={'size': 8}, framealpha=1, fancybox=True)
# HICP (2Y)
a = axes[1, 1]
col_cyc = cycler('color', ['red']+['dimgrey']*spf_bal_HICP_2Y.shape[1])
mar_cyc = cycler('marker', ['.']+['']*spf_bal_HICP_2Y.shape[1])
a.set_prop_cycle(col_cyc+mar_cyc)
spf_bal_HICP_2Y.plot(ax=a, linewidth=0.5)
a.set_title('Harmonised Inflation (2 Year Horizon)')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend([Line2D([0], [0], color='red', lw=1, marker='.'),
          Line2D([0], [0], color='dimgrey', lw=1)],
         ['HICP', 'Individual Forecasts'],
         loc=3, ncol=1, prop={'size': 8}, framealpha=1, fancybox=True)
# UNEM (1Y)
a = axes[2, 0]
col_cyc = cycler('color', ['red']+['dimgrey']*spf_bal_UNEM_1Y.shape[1])
mar_cyc = cycler('marker', ['.']+['']*spf_bal_UNEM_1Y.shape[1])
a.set_prop_cycle(col_cyc+mar_cyc)
spf_bal_UNEM_1Y.plot(ax=a, linewidth=0.5)
a.set_title('Unemployment Rate (1 Year Horizon)')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend([Line2D([0], [0], color='red', lw=1, marker='.'),
          Line2D([0], [0], color='dimgrey', lw=1)],
         ['UNEM', 'Individual Forecasts'],
         loc=3, ncol=1, prop={'size': 8}, framealpha=1, fancybox=True)
# UNEM (2Y)
a = axes[2, 1]
col_cyc = cycler('color', ['red']+['dimgrey']*spf_bal_UNEM_2Y.shape[1])
mar_cyc = cycler('marker', ['.']+['']*spf_bal_UNEM_2Y.shape[1])
a.set_prop_cycle(col_cyc+mar_cyc)
spf_bal_UNEM_2Y.plot(ax=a, linewidth=0.5)
a.set_title('Unemployment Rate (2 Year Horizon)')
a.set_xlabel('Time')
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend([Line2D([0], [0], color='red', lw=1, marker='.'),
          Line2D([0], [0], color='dimgrey', lw=1)],
         ['UNEM', 'Individual Forecasts'],
         loc=3, ncol=1, prop={'size': 8}, framealpha=1, fancybox=True)
# whole figure
fig.autofmt_xdate()
fig.savefig(fig_path + "spf_ind_fcts.pdf", bbox_inches='tight')

###############################
# INDIVIDUAL PERFORMANCE TABLE#
###############################

perf_stat = ["Mean", "Best Individual", "0.25 Quantile", "Median Individual",
             "0.75 Quantile", "Worst Individual"]*3+["Number of Forecasters"]
tickers = ['RGDP (1Y)', 'RGDP (2Y)', 'HICP (1Y)', 'HICP (2Y)',
           'UNEM (1Y)', 'UNEM (2Y)']

# RMSE of individual forecasters from SPF
rmse_RGDP_1Y = np.sqrt(np.mean(np.subtract(
        spf_bal_RGDP_1Y.values[:, 1:],
        spf_bal_RGDP_1Y.values[:, 0][:, np.newaxis])**2, axis=0))
rmse_RGDP_2Y = np.sqrt(np.mean(np.subtract(
        spf_bal_RGDP_2Y.values[:, 1:],
        spf_bal_RGDP_2Y.values[:, 0][:, np.newaxis])**2, axis=0))
rmse_HICP_1Y = np.sqrt(np.mean(np.subtract(
        spf_bal_HICP_1Y.values[:, 1:],
        spf_bal_HICP_1Y.values[:, 0][:, np.newaxis])**2, axis=0))
rmse_HICP_2Y = np.sqrt(np.mean(np.subtract(
        spf_bal_HICP_2Y.values[:, 1:],
        spf_bal_HICP_2Y.values[:, 0][:, np.newaxis])**2, axis=0))
rmse_UNEM_1Y = np.sqrt(np.mean(np.subtract(
        spf_bal_UNEM_1Y.values[:, 1:],
        spf_bal_UNEM_1Y.values[:, 0][:, np.newaxis])**2, axis=0))
rmse_UNEM_2Y = np.sqrt(np.mean(np.subtract(
        spf_bal_UNEM_2Y.values[:, 1:],
        spf_bal_UNEM_2Y.values[:, 0][:, np.newaxis])**2, axis=0))

# MAE of individual forecasters from SPF
mae_RGDP_1Y = np.mean(np.abs(np.subtract(
        spf_bal_RGDP_1Y.values[:, 1:],
        spf_bal_RGDP_1Y.values[:, 0][:, np.newaxis])), axis=0)
mae_RGDP_2Y = np.mean(np.abs(np.subtract(
        spf_bal_RGDP_2Y.values[:, 1:],
        spf_bal_RGDP_2Y.values[:, 0][:, np.newaxis])), axis=0)
mae_HICP_1Y = np.mean(np.abs(np.subtract(
        spf_bal_HICP_1Y.values[:, 1:],
        spf_bal_HICP_1Y.values[:, 0][:, np.newaxis])), axis=0)
mae_HICP_2Y = np.mean(np.abs(np.subtract(
        spf_bal_HICP_2Y.values[:, 1:],
        spf_bal_HICP_2Y.values[:, 0][:, np.newaxis])), axis=0)
mae_UNEM_1Y = np.mean(np.abs(np.subtract(
        spf_bal_UNEM_1Y.values[:, 1:],
        spf_bal_UNEM_1Y.values[:, 0][:, np.newaxis])), axis=0)
mae_UNEM_2Y = np.mean(np.abs(np.subtract(
        spf_bal_UNEM_2Y.values[:, 1:],
        spf_bal_UNEM_2Y.values[:, 0][:, np.newaxis])), axis=0)

# MAPE of individual forecasters from SPF

mape_RGDP_1Y = np.mean(np.absolute(np.subtract(
        spf_bal_RGDP_1Y.values[:, 1:],
        spf_bal_RGDP_1Y.values[:, 0][:, np.newaxis])[
        spf_bal_RGDP_1Y.values[:, 0] != 0, :] / spf_bal_RGDP_1Y.values[:, 0][
                :, np.newaxis][spf_bal_RGDP_1Y.values[:, 0] != 0, :]*100), 0)
mape_RGDP_2Y = np.mean(np.absolute(np.subtract(
        spf_bal_RGDP_2Y.values[:, 1:],
        spf_bal_RGDP_2Y.values[:, 0][:, np.newaxis])[
        spf_bal_RGDP_2Y.values[:, 0] != 0, :] / spf_bal_RGDP_2Y.values[:, 0][
                :, np.newaxis][spf_bal_RGDP_2Y.values[:, 0] != 0, :]*100), 0)
mape_HICP_1Y = np.mean(np.absolute(np.subtract(
        spf_bal_HICP_1Y.values[:, 1:],
        spf_bal_HICP_1Y.values[:, 0][:, np.newaxis])[
        spf_bal_HICP_1Y.values[:, 0] != 0, :] / spf_bal_HICP_1Y.values[:, 0][
                :, np.newaxis][spf_bal_HICP_1Y.values[:, 0] != 0, :]*100), 0)
mape_HICP_2Y = np.mean(np.absolute(np.subtract(
        spf_bal_HICP_2Y.values[:, 1:],
        spf_bal_HICP_2Y.values[:, 0][:, np.newaxis])[
        spf_bal_HICP_2Y.values[:, 0] != 0, :] / spf_bal_HICP_2Y.values[:, 0][
                :, np.newaxis][spf_bal_HICP_2Y.values[:, 0] != 0, :]*100), 0)
mape_UNEM_1Y = np.mean(np.absolute(np.subtract(
        spf_bal_UNEM_1Y.values[:, 1:],
        spf_bal_UNEM_1Y.values[:, 0][:, np.newaxis])[
        spf_bal_UNEM_1Y.values[:, 0] != 0, :] / spf_bal_UNEM_1Y.values[:, 0][
                :, np.newaxis][spf_bal_UNEM_1Y.values[:, 0] != 0, :]*100), 0)
mape_UNEM_2Y = np.mean(np.absolute(np.subtract(
        spf_bal_UNEM_2Y.values[:, 1:],
        spf_bal_UNEM_2Y.values[:, 0][:, np.newaxis])[
        spf_bal_UNEM_2Y.values[:, 0] != 0, :] / spf_bal_UNEM_2Y.values[:, 0][
                :, np.newaxis][spf_bal_UNEM_2Y.values[:, 0] != 0, :]*100), 0)

# table of individual forecast performance
spf_ind_perf_stat = pd.DataFrame(data=np.full((19, 6), np.nan, dtype=np.float),
                                 columns=tickers,
                                 index=perf_stat)
# RMSE
spf_ind_perf_stat.iloc[0, :] = np.array([
        np.mean(rmse_RGDP_1Y, axis=0),
        np.mean(rmse_RGDP_2Y, axis=0),
        np.mean(rmse_HICP_1Y, axis=0),
        np.mean(rmse_HICP_2Y, axis=0),
        np.mean(rmse_UNEM_1Y, axis=0),
        np.mean(rmse_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[1, :] = np.array([
        np.min(rmse_RGDP_1Y, axis=0),
        np.min(rmse_RGDP_2Y, axis=0),
        np.min(rmse_HICP_1Y, axis=0),
        np.min(rmse_HICP_2Y, axis=0),
        np.min(rmse_UNEM_1Y, axis=0),
        np.min(rmse_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[2, :] = np.array([
        np.percentile(rmse_RGDP_1Y, q=25, axis=0),
        np.percentile(rmse_RGDP_2Y, q=25, axis=0),
        np.percentile(rmse_HICP_1Y, q=25, axis=0),
        np.percentile(rmse_HICP_2Y, q=25, axis=0),
        np.percentile(rmse_UNEM_1Y, q=25, axis=0),
        np.percentile(rmse_UNEM_2Y, q=25, axis=0),
        ])
spf_ind_perf_stat.iloc[3, :] = np.array([
        np.median(rmse_RGDP_1Y, axis=0),
        np.median(rmse_RGDP_2Y, axis=0),
        np.median(rmse_HICP_1Y, axis=0),
        np.median(rmse_HICP_2Y, axis=0),
        np.median(rmse_UNEM_1Y, axis=0),
        np.median(rmse_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[4, :] = np.array([
        np.percentile(rmse_RGDP_1Y, q=75, axis=0),
        np.percentile(rmse_RGDP_2Y, q=75, axis=0),
        np.percentile(rmse_HICP_1Y, q=75, axis=0),
        np.percentile(rmse_HICP_2Y, q=75, axis=0),
        np.percentile(rmse_UNEM_1Y, q=75, axis=0),
        np.percentile(rmse_UNEM_2Y, q=75, axis=0),
        ])
spf_ind_perf_stat.iloc[5, :] = np.array([
        np.max(rmse_RGDP_1Y, axis=0),
        np.max(rmse_RGDP_2Y, axis=0),
        np.max(rmse_HICP_1Y, axis=0),
        np.max(rmse_HICP_2Y, axis=0),
        np.max(rmse_UNEM_1Y, axis=0),
        np.max(rmse_UNEM_2Y, axis=0),
        ])

# MAE
spf_ind_perf_stat.iloc[6, :] = np.array([
        np.mean(mae_RGDP_1Y, axis=0),
        np.mean(mae_RGDP_2Y, axis=0),
        np.mean(mae_HICP_1Y, axis=0),
        np.mean(mae_HICP_2Y, axis=0),
        np.mean(mae_UNEM_1Y, axis=0),
        np.mean(mae_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[7, :] = np.array([
        np.min(mae_RGDP_1Y, axis=0),
        np.min(mae_RGDP_2Y, axis=0),
        np.min(mae_HICP_1Y, axis=0),
        np.min(mae_HICP_2Y, axis=0),
        np.min(mae_UNEM_1Y, axis=0),
        np.min(mae_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[8, :] = np.array([
        np.percentile(mae_RGDP_1Y, q=25, axis=0),
        np.percentile(mae_RGDP_2Y, q=25, axis=0),
        np.percentile(mae_HICP_1Y, q=25, axis=0),
        np.percentile(mae_HICP_2Y, q=25, axis=0),
        np.percentile(mae_UNEM_1Y, q=25, axis=0),
        np.percentile(mae_UNEM_2Y, q=25, axis=0),
        ])
spf_ind_perf_stat.iloc[9, :] = np.array([
        np.median(mae_RGDP_1Y, axis=0),
        np.median(mae_RGDP_2Y, axis=0),
        np.median(mae_HICP_1Y, axis=0),
        np.median(mae_HICP_2Y, axis=0),
        np.median(mae_UNEM_1Y, axis=0),
        np.median(mae_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[10, :] = np.array([
        np.percentile(mae_RGDP_1Y, q=75, axis=0),
        np.percentile(mae_RGDP_2Y, q=75, axis=0),
        np.percentile(mae_HICP_1Y, q=75, axis=0),
        np.percentile(mae_HICP_2Y, q=75, axis=0),
        np.percentile(mae_UNEM_1Y, q=75, axis=0),
        np.percentile(mae_UNEM_2Y, q=75, axis=0),
        ])
spf_ind_perf_stat.iloc[11, :] = np.array([
        np.max(mae_RGDP_1Y, axis=0),
        np.max(mae_RGDP_2Y, axis=0),
        np.max(mae_HICP_1Y, axis=0),
        np.max(mae_HICP_2Y, axis=0),
        np.max(mae_UNEM_1Y, axis=0),
        np.max(mae_UNEM_2Y, axis=0),
        ])

# MAPE
spf_ind_perf_stat.iloc[12, :] = np.array([
        np.mean(mape_RGDP_1Y, axis=0),
        np.mean(mape_RGDP_2Y, axis=0),
        np.mean(mape_HICP_1Y, axis=0),
        np.mean(mape_HICP_2Y, axis=0),
        np.mean(mape_UNEM_1Y, axis=0),
        np.mean(mape_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[13, :] = np.array([
        np.min(mape_RGDP_1Y, axis=0),
        np.min(mape_RGDP_2Y, axis=0),
        np.min(mape_HICP_1Y, axis=0),
        np.min(mape_HICP_2Y, axis=0),
        np.min(mape_UNEM_1Y, axis=0),
        np.min(mape_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[14, :] = np.array([
        np.percentile(mape_RGDP_1Y, q=25, axis=0),
        np.percentile(mape_RGDP_2Y, q=25, axis=0),
        np.percentile(mape_HICP_1Y, q=25, axis=0),
        np.percentile(mape_HICP_2Y, q=25, axis=0),
        np.percentile(mape_UNEM_1Y, q=25, axis=0),
        np.percentile(mape_UNEM_2Y, q=25, axis=0),
        ])
spf_ind_perf_stat.iloc[15, :] = np.array([
        np.median(mape_RGDP_1Y, axis=0),
        np.median(mape_RGDP_2Y, axis=0),
        np.median(mape_HICP_1Y, axis=0),
        np.median(mape_HICP_2Y, axis=0),
        np.median(mape_UNEM_1Y, axis=0),
        np.median(mape_UNEM_2Y, axis=0),
        ])
spf_ind_perf_stat.iloc[16, :] = np.array([
        np.percentile(mape_RGDP_1Y, q=75, axis=0),
        np.percentile(mape_RGDP_2Y, q=75, axis=0),
        np.percentile(mape_HICP_1Y, q=75, axis=0),
        np.percentile(mape_HICP_2Y, q=75, axis=0),
        np.percentile(mape_UNEM_1Y, q=75, axis=0),
        np.percentile(mape_UNEM_2Y, q=75, axis=0),
        ])
spf_ind_perf_stat.iloc[17, :] = np.array([
        np.max(mape_RGDP_1Y, axis=0),
        np.max(mape_RGDP_2Y, axis=0),
        np.max(mape_HICP_1Y, axis=0),
        np.max(mape_HICP_2Y, axis=0),
        np.max(mape_UNEM_1Y, axis=0),
        np.max(mape_UNEM_2Y, axis=0),
        ])

# number of forecasters
spf_ind_perf_stat.iloc[18, :] = np.array([
        rmse_RGDP_1Y.shape[0],
        rmse_RGDP_2Y.shape[0],
        rmse_HICP_1Y.shape[0],
        rmse_HICP_2Y.shape[0],
        rmse_UNEM_1Y.shape[0],
        rmse_UNEM_2Y.shape[0],
        ])


# create table object
tabl = Table()
tabl.add_caption("Forecast performance (measured in terms of RMSE, MAE and MAPE) of indindividual forecasters from the ECB SPF for the target macroeconomic variables and horizons")
tabl.append(NoEscape('\label{tab: spf_ind_perf}'))
# create tabular object
tabr = Tabular(table_spec="c|l|cc|cc|cc")
tabr.add_hline()
tabr.add_hline()
# header row

tabr.add_row((MultiRow(2, data="Measure"), MultiRow(2, data="Performance"),
              MultiColumn(2, align='|c|', data="RGDP"),
              MultiColumn(2, align='|c|', data="HICP"),
              MultiColumn(2, align='|c', data="UNEM")))
tabr.add_hline(start=3, end=4, cmidruleoption="lr")
tabr.add_hline(start=5, end=6, cmidruleoption="lr")
tabr.add_hline(start=7, end=8, cmidruleoption="lr")
tabr.add_row([""]*2 + ["1Y", "2Y"] + ["1Y", "2Y"] + ["1Y", "2Y"])
tabr.add_hline()
# fill in the rows of tabular
tabr.add_row([MultiRow(6, data="RMSE")] + [perf_stat[0]] + [
            "{:.2f}".format(item) for item in spf_ind_perf_stat.iloc[0, :]])

for i in range(1, 6):
    tabr.add_row([""] + [perf_stat[i]] + [
            "{:.2f}".format(item) for item in spf_ind_perf_stat.iloc[i, :]])

tabr.add_hline()
tabr.add_row([MultiRow(6, data="MAE")] + [perf_stat[6]] + [
            "{:.2f}".format(item) for item in spf_ind_perf_stat.iloc[6, :]])

for i in range(7, 12):
    tabr.add_row([""] + [perf_stat[i]] + [
            "{:.2f}".format(item) for item in spf_ind_perf_stat.iloc[i, :]])

tabr.add_hline()
tabr.add_row([MultiRow(6, data="MAPE")] + [perf_stat[12]] + [
            "{:.2f}".format(item) for item in spf_ind_perf_stat.iloc[12, :]])

for i in range(13, 18):
    tabr.add_row([""] + [perf_stat[i]] + [
            "{:.2f}".format(item) for item in spf_ind_perf_stat.iloc[i, :]])

tabr.add_hline()

tabr.add_row([MultiColumn(2, align='c|', data=perf_stat[18])] + [
            "{:.0f}".format(item) for item in spf_ind_perf_stat.iloc[18, :]])
# end of table
tabr.add_hline()
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "spf_ind_perf")

###############
# END OF FILE #
###############
