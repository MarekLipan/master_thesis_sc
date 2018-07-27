"""
ECB Survey of Professional Forecasters

ANALYSIS SCRIPT

This script is used to perform the main analysis, i.e. train and test models.

"""

import numpy as np
import pandas as pd
import forecast_tables as ft
import random
import combination_methods as cm
import time
from pylatex import Table, Tabular, MultiColumn, MultiRow
from pylatex.utils import NoEscape

# set the seed for replicability of results
random.seed(444)
np.random.seed(444)


# testing
#w=100
#for i in range(ind_fcts_1_TU.shape[0]-w):
#    start_time = time.time()
#    df_train = ind_fcts_1_TU.iloc[i:(w+i), :]
#    df_test = ind_fcts_1_TU.iloc[(w+i):(w+i+1), 1:]
#    ############################
#    fcts = pd.concat([
#        cm.Equal_Weights(df_test),
#        cm.Bates_Granger_1(df_train, df_test),
#        cm.Bates_Granger_2(df_train, df_test),
#        cm.Bates_Granger_3(df_train, df_test, alpha=0.6),
#        cm.Bates_Granger_4(df_train, df_test, W=1.5),
#        cm.Bates_Granger_5(df_train, df_test, W=1.5),
#        cm.Granger_Ramanathan_1(df_train, df_test),
#        cm.Granger_Ramanathan_2(df_train, df_test),
#        cm.Granger_Ramanathan_3(df_train, df_test),
#        cm.AFTER(df_train, df_test, lambd=0.15),
#        cm.Median_Forecast(df_test),
#        cm.Trimmed_Mean_Forecast(df_test, alpha=0.05),
#        cm.PEW(df_train, df_test),
#        cm.Principal_Component_Forecast(df_train, df_test, "single"),
#        cm.Principal_Component_Forecast(df_train, df_test, "AIC"),
#        cm.Principal_Component_Forecast(df_train, df_test, "BIC"),
#        cm.Empirical_Bayes_Estimator(df_train, df_test),
#        cm.Kappa_Shrinkage(df_train, df_test, kappa=0.5),
#        cm.Two_Step_Egalitarian_LASSO(df_train, df_test, k_cv=5,
#                                      grid_l=-20, grid_h=2, grid_size=20),
#        cm.BMA_Marginal_Likelihood_exh(df_train, df_test),
#        cm.BMA_Predictive_Likelihood_exh(df_train, df_test, l_share=0.1),
#        cm.ANN(df_train, df_test),
#        cm.EP_NN(df_train, df_test, sigma=0.05, gen=200, n=16),
#        cm.Bagging(df_train, df_test, B=500, threshold=1.28),
#        cm.Componentwise_Boosting(df_train, df_test, nu=0.1),
#        cm.AdaBoost(df_train, df_test, phi=0.1),
#        cm.cAPM_Constant(df_train, df_test, MaxRPT_r1=0.9, MaxRPT=0.01,
#                         no_rounds=10),
#        cm.cAPM_Q_learning(df_train, df_test, MinRPT=0.0001, MaxRPT_r1=0.9,
#                           MaxRPT=0.01, alpha=0.7, no_rounds=10),
#        cm.MK(df_train, df_test)
#    ], axis=1).values[0]
#    ############################
#    end_time = time.time()
#    print(str(i) + ": " + str(end_time-start_time))

#for i in range(ind_fcts_1_TU.shape[0]-w):
#    start_time = time.time()
#    df_train = ind_fcts_1_TU.iloc[i:(w+i), :]
#    df_test = ind_fcts_1_TU.iloc[(w+i):(w+i+1), 1:]
#    ############################
#    pred = cm.BMA_Predictive_Likelihood_exh(df_train, df_test, l_share=0.1)
#    ############################
#    end_time = time.time()
#    print(str(i) + ": " + str(end_time-start_time))
#    print("prediction: " + str(pred.values[0][0]))


# create accuracy tables
#acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=500,
#                                        proc="single",
#                                        df_name="ind_fcts_1_TU_"+str(w))

#for w in [100, 200, 500]:
#
#    # load the data and compute accuracy measures
#    acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_1_TU_"+str(w))
#
#    acc_table_ind_fcts_5_TU = ft.create_acc_table(df=ind_fcts_5_TU, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_5_TU_"+str(w))
#
#    acc_table_ind_fcts_22_TU = ft.create_acc_table(df=ind_fcts_22_TU, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_22_TU_"+str(w))
#
#    acc_table_ind_fcts_1_FV = ft.create_acc_table(df=ind_fcts_1_FV, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_1_FV_"+str(w))
#
#    acc_table_ind_fcts_5_FV = ft.create_acc_table(df=ind_fcts_5_FV, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_5_FV_"+str(w))
#
#    acc_table_ind_fcts_22_FV = ft.create_acc_table(df=ind_fcts_22_FV, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_22_FV_"+str(w))
#
#    acc_table_ind_fcts_1_TY = ft.create_acc_table(df=ind_fcts_1_TY, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_1_TY_"+str(w))
#
#    acc_table_ind_fcts_5_TY = ft.create_acc_table(df=ind_fcts_5_TY, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_5_TY_"+str(w))
#
#    acc_table_ind_fcts_22_TY = ft.create_acc_table(df=ind_fcts_22_TY, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_22_TY_"+str(w))
#
#    acc_table_ind_fcts_1_US = ft.create_acc_table(df=ind_fcts_1_US, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_1_US_"+str(w))
#
#    acc_table_ind_fcts_5_US = ft.create_acc_table(df=ind_fcts_5_US, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_5_US_"+str(w))
#
#    acc_table_ind_fcts_22_US = ft.create_acc_table(df=ind_fcts_22_US, w=w,
#                                            proc="multiple",
#                                            df_name="ind_fcts_22_US_"+str(w))
#
#    # export accuracy tables to tex
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_1_TU,
#                     cap="Combined 1-step-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_1_TU_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_5_TU,
#                     cap="Combined 5-steps-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_5_TU_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_22_TU,
#                     cap="Combined 22-steps-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_22_TU_"+str(w),
#                     r=6)
#
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_1_FV,
#                     cap="Combined 1-step-ahead forecasts of the realized volatility of log-returns of FV (5 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_1_FV_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_5_FV,
#                     cap="Combined 5-steps-ahead forecasts of the realized volatility of log-returns of FV (5 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_5_FV_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_22_FV,
#                     cap="Combined 22-steps-ahead forecasts of the realized volatility of log-returns of FV (5 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_22_FV_"+str(w),
#                     r=6)
#
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_1_TY,
#                     cap="Combined 1-step-ahead forecasts of the realized volatility of log-returns of TY (10 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_1_TY_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_5_TY,
#                     cap="Combined 5-steps-ahead forecasts of the realized volatility of log-returns of TY (10 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_5_TY_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_22_TY,
#                     cap="Combined 22-steps-ahead forecasts of the realized volatility of log-returns of TY (10 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_22_TY_"+str(w),
#                     r=6)
#
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_1_US,
#                     cap="Combined 1-step-ahead forecasts of the realized volatility of log-returns of US (30 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_1_US_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_5_US,
#                     cap="Combined 5-steps-ahead forecasts of the realized volatility of log-returns of US (30 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_5_US_"+str(w),
#                     r=6)
#    ft.gen_tex_table(tbl=acc_table_ind_fcts_22_US,
#                     cap="Combined 22-steps-ahead forecasts of the realized volatility of log-returns of US (30 Year) futures (w="+str(w)+")",
#                     file_name="ind_fcts_22_US_"+str(w),
#                     r=6)

##########
# OUTPUTS#
##########
# path to where the figures and tables are stored
fig_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Figures/"
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"

# load for index
acc_table_ind_fcts_ind = ft.create_acc_table(df=ind_fcts_1_TU, w=100,
                                             proc="multiple",
                                             df_name="ind_fcts_1_TU_"+str(100))
##############
# TU (2 Year)#
##############
# load and concatenate the tables together
acc_table_ind_fcts_TU = pd.concat(
        [ft.create_acc_table(df=ind_fcts_1_TU, w=100,proc="multiple",df_name="ind_fcts_1_TU_"+str(100)),
         ft.create_acc_table(df=ind_fcts_1_TU, w=200,proc="multiple",df_name="ind_fcts_1_TU_"+str(200)),
         ft.create_acc_table(df=ind_fcts_1_TU, w=500,proc="multiple",df_name="ind_fcts_1_TU_"+str(500)),

         ft.create_acc_table(df=ind_fcts_5_TU, w=100,proc="multiple",df_name="ind_fcts_5_TU_"+str(100)),
         ft.create_acc_table(df=ind_fcts_5_TU, w=200,proc="multiple",df_name="ind_fcts_5_TU_"+str(200)),
         ft.create_acc_table(df=ind_fcts_5_TU, w=500,proc="multiple",df_name="ind_fcts_5_TU_"+str(500)),

         ft.create_acc_table(df=ind_fcts_22_TU, w=100,proc="multiple",df_name="ind_fcts_22_TU_"+str(100)),
         ft.create_acc_table(df=ind_fcts_22_TU, w=200,proc="multiple",df_name="ind_fcts_22_TU_"+str(200)),
         ft.create_acc_table(df=ind_fcts_22_TU, w=500,proc="multiple",df_name="ind_fcts_22_TU_"+str(500))
         ], axis=1, join_axes=[acc_table_ind_fcts_ind.index])

# drop MAE for space reasons
acc_table_ind_fcts_TU = acc_table_ind_fcts_TU.drop(columns='MAE')

# scale the RMSE
acc_table_ind_fcts_TU.loc[:,'RMSE'] *= 10000

# create table object
tabl = Table()
tabl.add_caption("Performance of forecast combinations, trained on a rolling window of length w, of inidividual h-steps-ahead forecasts of realized volatility of log-returns of U.S. Treasury futures: TU (2 Year)")
tabl.append(NoEscape('\label{tab: bond_comb_TU}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 9*"|cc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"), MultiRow(3, data="Forecast Combination Method"),
                  MultiColumn(6, align='c', data="h = 1"),
                  MultiColumn(6, align='|c', data="h = 5"),
                  MultiColumn(6, align='|c', data="h = 22")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(("", "",
                  MultiColumn(2, align='c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(2*[""] + 9*["RMSE", "MAPE"])
tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [acc_table_ind_fcts_TU.index[0]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[0, :]])
for i in range(1, 13):
    tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [acc_table_ind_fcts_TU.index[13]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[13, :]])
for i in range(14, 16):
    tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [acc_table_ind_fcts_TU.index[16]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[16, :]])
for i in range(17, 19):
    tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [acc_table_ind_fcts_TU.index[19]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[19, :]])
for i in range(20, 21):
    tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [acc_table_ind_fcts_TU.index[21]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[21, :]])
for i in range(22, 26):
        tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
                "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [acc_table_ind_fcts_TU.index[26]] + [
         "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[26, :]])
for i in range(27, 29):
    tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

tabr.add_hline()
for i in range(29, 32):
    tabr.add_row([""] + [acc_table_ind_fcts_TU.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TU.iloc[i, :]])

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_comb_TU")

##############
# FV (5 Year)#
##############
# load and concatenate the tables together
acc_table_ind_fcts_FV = pd.concat(
        [ft.create_acc_table(df=ind_fcts_1_FV, w=100,proc="multiple",df_name="ind_fcts_1_FV_"+str(100)),
         ft.create_acc_table(df=ind_fcts_1_FV, w=200,proc="multiple",df_name="ind_fcts_1_FV_"+str(200)),
         ft.create_acc_table(df=ind_fcts_1_FV, w=500,proc="multiple",df_name="ind_fcts_1_FV_"+str(500)),

         ft.create_acc_table(df=ind_fcts_5_FV, w=100,proc="multiple",df_name="ind_fcts_5_FV_"+str(100)),
         ft.create_acc_table(df=ind_fcts_5_FV, w=200,proc="multiple",df_name="ind_fcts_5_FV_"+str(200)),
         ft.create_acc_table(df=ind_fcts_5_FV, w=500,proc="multiple",df_name="ind_fcts_5_FV_"+str(500)),

         ft.create_acc_table(df=ind_fcts_22_FV, w=100,proc="multiple",df_name="ind_fcts_22_FV_"+str(100)),
         ft.create_acc_table(df=ind_fcts_22_FV, w=200,proc="multiple",df_name="ind_fcts_22_FV_"+str(200)),
         ft.create_acc_table(df=ind_fcts_22_FV, w=500,proc="multiple",df_name="ind_fcts_22_FV_"+str(500))
         ], axis=1, join_axes=[acc_table_ind_fcts_ind.index])

# drop MAE for space reasons
acc_table_ind_fcts_FV = acc_table_ind_fcts_FV.drop(columns='MAE')

# scale the RMSE
acc_table_ind_fcts_FV.loc[:,'RMSE'] *= 10000

# create table object
tabl = Table()
tabl.add_caption("Performance of forecast combinations, trained on a rolling window of length w, of inidividual h-steps-ahead forecasts of realized volatility of log-reFVrns of U.S. Treasury fuFVres: FV (5 Year)")
tabl.append(NoEscape('label{tab: bond_comb_FV}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 9*"|cc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"), MultiRow(3, data="Forecast Combination Method"),
                  MultiColumn(6, align='c', data="h = 1"),
                  MultiColumn(6, align='|c', data="h = 5"),
                  MultiColumn(6, align='|c', data="h = 22")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(("", "",
                  MultiColumn(2, align='c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(2*[""] + 9*["RMSE", "MAPE"])
tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [acc_table_ind_fcts_FV.index[0]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[0, :]])
for i in range(1, 13):
    tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [acc_table_ind_fcts_FV.index[13]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[13, :]])
for i in range(14, 16):
    tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [acc_table_ind_fcts_FV.index[16]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[16, :]])
for i in range(17, 19):
    tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [acc_table_ind_fcts_FV.index[19]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[19, :]])
for i in range(20, 21):
    tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [acc_table_ind_fcts_FV.index[21]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[21, :]])
for i in range(22, 26):
        tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
                "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [acc_table_ind_fcts_FV.index[26]] + [
         "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[26, :]])
for i in range(27, 29):
    tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

tabr.add_hline()
for i in range(29, 32):
    tabr.add_row([""] + [acc_table_ind_fcts_FV.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_FV.iloc[i, :]])

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_comb_FV")

##############
# TY (10 Year)#
##############
# load and concatenate the tables together
acc_table_ind_fcts_TY = pd.concat(
        [ft.create_acc_table(df=ind_fcts_1_TY, w=100,proc="multiple",df_name="ind_fcts_1_TY_"+str(100)),
         ft.create_acc_table(df=ind_fcts_1_TY, w=200,proc="multiple",df_name="ind_fcts_1_TY_"+str(200)),
         ft.create_acc_table(df=ind_fcts_1_TY, w=500,proc="multiple",df_name="ind_fcts_1_TY_"+str(500)),

         ft.create_acc_table(df=ind_fcts_5_TY, w=100,proc="multiple",df_name="ind_fcts_5_TY_"+str(100)),
         ft.create_acc_table(df=ind_fcts_5_TY, w=200,proc="multiple",df_name="ind_fcts_5_TY_"+str(200)),
         ft.create_acc_table(df=ind_fcts_5_TY, w=500,proc="multiple",df_name="ind_fcts_5_TY_"+str(500)),

         ft.create_acc_table(df=ind_fcts_22_TY, w=100,proc="multiple",df_name="ind_fcts_22_TY_"+str(100)),
         ft.create_acc_table(df=ind_fcts_22_TY, w=200,proc="multiple",df_name="ind_fcts_22_TY_"+str(200)),
         ft.create_acc_table(df=ind_fcts_22_TY, w=500,proc="multiple",df_name="ind_fcts_22_TY_"+str(500))
         ], axis=1, join_axes=[acc_table_ind_fcts_ind.index])

# drop MAE for space reasons
acc_table_ind_fcts_TY = acc_table_ind_fcts_TY.drop(columns='MAE')

# scale the RMSE
acc_table_ind_fcts_TY.loc[:,'RMSE'] *= 10000

# create table object
tabl = Table()
tabl.add_caption("Performance of forecast combinations, trained on a rolling window of length w, of inidividual h-steps-ahead forecasts of realized volatility of log-reTYrns of U.S. Treasury fuTYres: TY (10 Year)")
tabl.append(NoEscape('label{tab: bond_comb_TY}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 9*"|cc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"), MultiRow(3, data="Forecast Combination Method"),
                  MultiColumn(6, align='c', data="h = 1"),
                  MultiColumn(6, align='|c', data="h = 5"),
                  MultiColumn(6, align='|c', data="h = 22")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(("", "",
                  MultiColumn(2, align='c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(2*[""] + 9*["RMSE", "MAPE"])
tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [acc_table_ind_fcts_TY.index[0]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[0, :]])
for i in range(1, 13):
    tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [acc_table_ind_fcts_TY.index[13]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[13, :]])
for i in range(14, 16):
    tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [acc_table_ind_fcts_TY.index[16]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[16, :]])
for i in range(17, 19):
    tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [acc_table_ind_fcts_TY.index[19]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[19, :]])
for i in range(20, 21):
    tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [acc_table_ind_fcts_TY.index[21]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[21, :]])
for i in range(22, 26):
        tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
                "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [acc_table_ind_fcts_TY.index[26]] + [
         "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[26, :]])
for i in range(27, 29):
    tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

tabr.add_hline()
for i in range(29, 32):
    tabr.add_row([""] + [acc_table_ind_fcts_TY.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_TY.iloc[i, :]])

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_comb_TY")

##############
# US (30 Year)#
##############
# load and concatenate the tables together
acc_table_ind_fcts_US = pd.concat(
        [ft.create_acc_table(df=ind_fcts_1_US, w=100,proc="multiple",df_name="ind_fcts_1_US_"+str(100)),
         ft.create_acc_table(df=ind_fcts_1_US, w=200,proc="multiple",df_name="ind_fcts_1_US_"+str(200)),
         ft.create_acc_table(df=ind_fcts_1_US, w=500,proc="multiple",df_name="ind_fcts_1_US_"+str(500)),

         ft.create_acc_table(df=ind_fcts_5_US, w=100,proc="multiple",df_name="ind_fcts_5_US_"+str(100)),
         ft.create_acc_table(df=ind_fcts_5_US, w=200,proc="multiple",df_name="ind_fcts_5_US_"+str(200)),
         ft.create_acc_table(df=ind_fcts_5_US, w=500,proc="multiple",df_name="ind_fcts_5_US_"+str(500)),

         ft.create_acc_table(df=ind_fcts_22_US, w=100,proc="multiple",df_name="ind_fcts_22_US_"+str(100)),
         ft.create_acc_table(df=ind_fcts_22_US, w=200,proc="multiple",df_name="ind_fcts_22_US_"+str(200)),
         ft.create_acc_table(df=ind_fcts_22_US, w=500,proc="multiple",df_name="ind_fcts_22_US_"+str(500))
         ], axis=1, join_axes=[acc_table_ind_fcts_ind.index])

# drop MAE for space reasons
acc_table_ind_fcts_US = acc_table_ind_fcts_US.drop(columns='MAE')

# scale the RMSE
acc_table_ind_fcts_US.loc[:,'RMSE'] *= 10000

# create table object
tabl = Table()
tabl.add_caption("Performance of forecast combinations, trained on a rolling window of length w, of inidividual h-steps-ahead forecasts of realized volatility of log-reUSrns of U.S. Treasury fuUSres: US (30 Year)")
tabl.append(NoEscape('label{tab: bond_comb_US}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 9*"|cc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"), MultiRow(3, data="Forecast Combination Method"),
                  MultiColumn(6, align='c', data="h = 1"),
                  MultiColumn(6, align='|c', data="h = 5"),
                  MultiColumn(6, align='|c', data="h = 22")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(("", "",
                  MultiColumn(2, align='c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500"),
                  MultiColumn(2, align='|c', data="w = 100"),
                  MultiColumn(2, align='|c', data="w = 200"),
                  MultiColumn(2, align='|c', data="w = 500")))
tabr.add_hline(start=3, end=20, cmidruleoption="lr")
tabr.add_row(2*[""] + 9*["RMSE", "MAPE"])
tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [acc_table_ind_fcts_US.index[0]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[0, :]])
for i in range(1, 13):
    tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [acc_table_ind_fcts_US.index[13]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[13, :]])
for i in range(14, 16):
    tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [acc_table_ind_fcts_US.index[16]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[16, :]])
for i in range(17, 19):
    tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [acc_table_ind_fcts_US.index[19]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[19, :]])
for i in range(20, 21):
    tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
            "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [acc_table_ind_fcts_US.index[21]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[21, :]])
for i in range(22, 26):
        tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
                "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [acc_table_ind_fcts_US.index[26]] + [
         "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[26, :]])
for i in range(27, 29):
    tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

tabr.add_hline()
for i in range(29, 32):
    tabr.add_row([""] + [acc_table_ind_fcts_US.index[i]] + [
        "{:.2f}".format(item) for item in acc_table_ind_fcts_US.iloc[i, :]])

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tabl.generate_tex(tab_path + "bond_comb_US")

# END OF FILE
