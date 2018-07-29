"""
RANK TABLE

This script is used to create and export table which displayes aggregated
average ranks among the forecast combination methods on all datasets.

"""

import pandas as pd
import numpy as np
import accuracy_measures as am
import forecast_tables as ft
from pylatex import Table, Tabular, MultiColumn, MultiRow
from pylatex.utils import NoEscape

#############
# RANK TABLE#
#############

# prepare the data
# ECB SPF
rank_RGDP_1Y = np.mean((
        am.rank_methods(ft.create_acc_table(df=spf_bal_RGDP_1Y, w=25, proc="multiple", df_name="spf_bal_RGDP_1Y_"+str(25))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_RGDP_1Y, w=35, proc="multiple", df_name="spf_bal_RGDP_1Y_"+str(35))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_RGDP_1Y, w=45, proc="multiple", df_name="spf_bal_RGDP_1Y_"+str(45)))
        ), axis=0)

rank_RGDP_2Y = np.mean((
        am.rank_methods(ft.create_acc_table(df=spf_bal_RGDP_2Y, w=25, proc="multiple", df_name="spf_bal_RGDP_2Y_"+str(25))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_RGDP_2Y, w=35, proc="multiple", df_name="spf_bal_RGDP_2Y_"+str(35))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_RGDP_2Y, w=45, proc="multiple", df_name="spf_bal_RGDP_2Y_"+str(45)))
        ), axis=0)

rank_HICP_1Y = np.mean((
        am.rank_methods(ft.create_acc_table(df=spf_bal_HICP_1Y, w=25, proc="multiple", df_name="spf_bal_HICP_1Y_"+str(25))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_HICP_1Y, w=35, proc="multiple", df_name="spf_bal_HICP_1Y_"+str(35))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_HICP_1Y, w=45, proc="multiple", df_name="spf_bal_HICP_1Y_"+str(45)))
        ), axis=0)

rank_HICP_2Y = np.mean((
        am.rank_methods(ft.create_acc_table(df=spf_bal_HICP_2Y, w=25, proc="multiple", df_name="spf_bal_HICP_2Y_"+str(25))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_HICP_2Y, w=35, proc="multiple", df_name="spf_bal_HICP_2Y_"+str(35))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_HICP_2Y, w=45, proc="multiple", df_name="spf_bal_HICP_2Y_"+str(45)))
        ), axis=0)

rank_UNEM_1Y = np.mean((
        am.rank_methods(ft.create_acc_table(df=spf_bal_UNEM_1Y, w=25, proc="multiple", df_name="spf_bal_UNEM_1Y_"+str(25))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_UNEM_1Y, w=35, proc="multiple", df_name="spf_bal_UNEM_1Y_"+str(35))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_UNEM_1Y, w=45, proc="multiple", df_name="spf_bal_UNEM_1Y_"+str(45)))
        ), axis=0)

rank_UNEM_2Y = np.mean((
        am.rank_methods(ft.create_acc_table(df=spf_bal_UNEM_2Y, w=25, proc="multiple", df_name="spf_bal_UNEM_2Y_"+str(25))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_UNEM_2Y, w=35, proc="multiple", df_name="spf_bal_UNEM_2Y_"+str(35))),
        am.rank_methods(ft.create_acc_table(df=spf_bal_UNEM_2Y, w=45, proc="multiple", df_name="spf_bal_UNEM_2Y_"+str(45)))
        ), axis=0)

rank_ECB_SPF = np.mean((rank_RGDP_1Y, rank_RGDP_2Y,
                        rank_HICP_1Y, rank_HICP_2Y,
                        rank_UNEM_1Y, rank_UNEM_2Y), axis=0)

# RVOL
rank_1_TU = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_TU, w=100,proc="multiple",df_name="ind_fcts_1_TU_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_TU, w=200, proc="multiple", df_name="ind_fcts_1_TU_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_TU, w=500, proc="multiple", df_name="ind_fcts_1_TU_"+str(500)))
        ), axis=0)

rank_1_TY = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_TY, w=100,proc="multiple",df_name="ind_fcts_1_TY_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_TY, w=200, proc="multiple", df_name="ind_fcts_1_TY_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_TY, w=500, proc="multiple", df_name="ind_fcts_1_TY_"+str(500)))
        ), axis=0)

rank_1_FV = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_FV, w=100,proc="multiple",df_name="ind_fcts_1_FV_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_FV, w=200, proc="multiple", df_name="ind_fcts_1_FV_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_FV, w=500, proc="multiple", df_name="ind_fcts_1_FV_"+str(500)))
        ), axis=0)

rank_1_US = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_US, w=100,proc="multiple",df_name="ind_fcts_1_US_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_US, w=200, proc="multiple", df_name="ind_fcts_1_US_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_1_US, w=500, proc="multiple", df_name="ind_fcts_1_US_"+str(500)))
        ), axis=0)

rank_5_TU = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_TU, w=100,proc="multiple",df_name="ind_fcts_5_TU_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_TU, w=200, proc="multiple", df_name="ind_fcts_5_TU_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_TU, w=500, proc="multiple", df_name="ind_fcts_5_TU_"+str(500)))
        ), axis=0)

rank_5_TY = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_TY, w=100,proc="multiple",df_name="ind_fcts_5_TY_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_TY, w=200, proc="multiple", df_name="ind_fcts_5_TY_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_TY, w=500, proc="multiple", df_name="ind_fcts_5_TY_"+str(500)))
        ), axis=0)

rank_5_FV = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_FV, w=100,proc="multiple",df_name="ind_fcts_5_FV_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_FV, w=200, proc="multiple", df_name="ind_fcts_5_FV_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_FV, w=500, proc="multiple", df_name="ind_fcts_5_FV_"+str(500)))
        ), axis=0)

rank_5_US = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_US, w=100,proc="multiple",df_name="ind_fcts_5_US_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_US, w=200, proc="multiple", df_name="ind_fcts_5_US_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_5_US, w=500, proc="multiple", df_name="ind_fcts_5_US_"+str(500)))
        ), axis=0)

rank_22_TU = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_TU, w=100,proc="multiple",df_name="ind_fcts_22_TU_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_TU, w=200, proc="multiple", df_name="ind_fcts_22_TU_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_TU, w=500, proc="multiple", df_name="ind_fcts_22_TU_"+str(500)))
        ), axis=0)

rank_22_TY = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_TY, w=100,proc="multiple",df_name="ind_fcts_22_TY_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_TY, w=200, proc="multiple", df_name="ind_fcts_22_TY_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_TY, w=500, proc="multiple", df_name="ind_fcts_22_TY_"+str(500)))
        ), axis=0)

rank_22_FV = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_FV, w=100,proc="multiple",df_name="ind_fcts_22_FV_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_FV, w=200, proc="multiple", df_name="ind_fcts_22_FV_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_FV, w=500, proc="multiple", df_name="ind_fcts_22_FV_"+str(500)))
        ), axis=0)

rank_22_US = np.mean((
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_US, w=100,proc="multiple",df_name="ind_fcts_22_US_"+str(100))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_US, w=200, proc="multiple", df_name="ind_fcts_22_US_"+str(200))),
        am.rank_methods(ft.create_acc_table(df=ind_fcts_22_US, w=500, proc="multiple", df_name="ind_fcts_22_US_"+str(500)))
        ), axis=0)

rank_RVOL = np.mean((
        rank_1_TU, rank_1_FV, rank_1_TY, rank_1_US,
        rank_5_TU, rank_5_FV, rank_5_TY, rank_5_US,
        rank_22_TU, rank_22_FV, rank_22_TY, rank_22_US), axis=0)

# TOTAL
rank_TOTAL = np.mean((rank_ECB_SPF, rank_RVOL), axis=0)


# put rank table together
comb_methods = ['Equal Weights', 'Bates-Granger (1)', 'Bates-Granger (2)',
       'Bates-Granger (3)', 'Bates-Granger (4)', 'Bates-Granger (5)',
       'Granger-Ramanathan (1)', 'Granger-Ramanathan (2)',
       'Granger-Ramanathan (3)', 'AFTER', 'Median Forecast',
       'Trimmed Mean Forecast', 'PEW', 'Principal Component Forecast',
       'Principal Component Forecast (AIC)',
       'Principal Component Forecast (BIC)', 'Empirical Bayes Estimator',
       'Kappa-Shrinkage', 'Two-Step Egalitarian LASSO',
       'BMA (Marginal Likelihood)', 'BMA (Predictive Likelihood)', 'ANN',
       'EP-NN', 'Bagging', 'Componentwise Boosting', 'AdaBoost',
       'c-APM (Constant)', 'c-APM (Q-learning)', 'Market for Kernels']

rank_table = np.concatenate([
        rank_TOTAL[:, np.newaxis],
        rank_ECB_SPF[:, np.newaxis],
        rank_RGDP_1Y[:, np.newaxis], rank_RGDP_2Y[:, np.newaxis],
        rank_HICP_1Y[:, np.newaxis], rank_HICP_2Y[:, np.newaxis],
        rank_UNEM_1Y[:, np.newaxis], rank_UNEM_2Y[:, np.newaxis],
        rank_RVOL[:, np.newaxis],
        rank_1_TU[:, np.newaxis], rank_1_FV[:, np.newaxis], rank_1_TY[:, np.newaxis], rank_1_US[:, np.newaxis],
        rank_5_TU[:, np.newaxis], rank_5_FV[:, np.newaxis], rank_5_TY[:, np.newaxis], rank_5_US[:, np.newaxis],
        rank_22_TU[:, np.newaxis], rank_22_FV[:, np.newaxis], rank_22_TY[:, np.newaxis], rank_22_US[:, np.newaxis]
        ], axis=1)

rank_table = pd.DataFrame(data=rank_table, index=comb_methods)





# create table object
tabl = Table()
tabl.add_caption("Average ranks of forecast combinations methods across all the datasets obtained by averaging the ranks based on RMSE, MAE and MAPE")
tabl.append(NoEscape('\label{tab: rank_table}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 2*"|c" + 3*"|cc" + "|c" + 3*"|cccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiRow(3, data="Total"),
              MultiColumn(7, align='c', data="ECB SPF"),
              MultiColumn(13, align='|c', data="U.S. Treasury Futures RVOL")))

tabr.add_hline(start=4, end=23, cmidruleoption="lr")

tabr.add_row(("", "", "",
              MultiRow(2, data="Subtotal"),
              MultiColumn(2, align='c', data="RGDP"),
              MultiColumn(2, align='|c', data="HICP"),
              MultiColumn(2, align='|c|', data="UNEM"),
              MultiRow(2, data="Subtotal"),
              MultiColumn(4, align='c', data="h = 1"),
              MultiColumn(4, align='|c', data="h = 5"),
              MultiColumn(4, align='|c', data="h = 22")))

tabr.add_hline(start=5, end=10, cmidruleoption="lr")
tabr.add_hline(start=12, end=23, cmidruleoption="lr")

tabr.add_row(4*[""] + 3*["1Y", "2Y"] + [""] + 3*["TU", "FV", "TY", "US"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [rank_table.index[0]] + [
        "{:.1f}".format(item) for item in rank_table.iloc[0, :]])
for i in range(1, 13):
    tabr.add_row([""] + [rank_table.index[i]] + [
            "{:.1f}".format(item) for item in rank_table.iloc[i, :]])

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [rank_table.index[13]] + [
        "{:.1f}".format(item) for item in rank_table.iloc[13, :]])
for i in range(14, 16):
    tabr.add_row([""] + [rank_table.index[i]] + [
            "{:.1f}".format(item) for item in rank_table.iloc[i, :]])

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [rank_table.index[16]] + [
        "{:.1f}".format(item) for item in rank_table.iloc[16, :]])
for i in range(17, 19):
    tabr.add_row([""] + [rank_table.index[i]] + [
            "{:.1f}".format(item) for item in rank_table.iloc[i, :]])

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [rank_table.index[19]] + [
        "{:.1f}".format(item) for item in rank_table.iloc[19, :]])
for i in range(20, 21):
    tabr.add_row([""] + [rank_table.index[i]] + [
            "{:.1f}".format(item) for item in rank_table.iloc[i, :]])

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [rank_table.index[21]] + [
        "{:.1f}".format(item) for item in rank_table.iloc[21, :]])
for i in range(22, 26):
        tabr.add_row([""] + [rank_table.index[i]] + [
                "{:.1f}".format(item) for item in rank_table.iloc[i, :]])

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [rank_table.index[26]] + [
         "{:.1f}".format(item) for item in rank_table.iloc[26, :]])
for i in range(27, 29):
    tabr.add_row([""] + [rank_table.index[i]] + [
        "{:.1f}".format(item) for item in rank_table.iloc[i, :]])

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "rank_table")

###############
# END OF FILE #
###############
