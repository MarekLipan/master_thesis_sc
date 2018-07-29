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
import accuracy_measures as am
import time
from pylatex import Table, Tabular, MultiColumn, MultiRow
from pylatex.utils import NoEscape
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler

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

###############################
# FORECAST COMBINATION FIGURES#
###############################
comb_methods = ['RVOL', 'Equal Weights', 'Bates-Granger (1)', 'Bates-Granger (2)',
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
data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"

# TU (2 Year)
# load accuracy tables
acc_table_1_100 = ft.create_acc_table(df=ind_fcts_1_TU, w=100,proc="multiple",df_name="ind_fcts_1_TU_"+str(100))
acc_table_1_500 = ft.create_acc_table(df=ind_fcts_1_TU, w=500,proc="multiple",df_name="ind_fcts_1_TU_"+str(500))

acc_table_5_100 = ft.create_acc_table(df=ind_fcts_5_TU, w=100,proc="multiple",df_name="ind_fcts_5_TU_"+str(100))
acc_table_5_500 = ft.create_acc_table(df=ind_fcts_5_TU, w=500,proc="multiple",df_name="ind_fcts_5_TU_"+str(500))

acc_table_22_100 = ft.create_acc_table(df=ind_fcts_22_TU, w=100,proc="multiple",df_name="ind_fcts_22_TU_"+str(100))
acc_table_22_500 = ft.create_acc_table(df=ind_fcts_22_TU, w=500,proc="multiple",df_name="ind_fcts_22_TU_"+str(500))

# prepare datasets for plotting
bond_comb_1_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_TU_"+str(100) + ".pkl")
bond_comb_1_100.insert(0, "RVOL", ind_fcts_1_TU.values[(-bond_comb_1_100.shape[0]):, 0])
bond_comb_1_100.columns = comb_methods
bond_comb_1_100.index = ind_fcts_1_TU.index[(-bond_comb_1_100.shape[0]):]
bond_comb_1_100 = bond_comb_1_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_100))+1)]
bond_comb_1_100 = bond_comb_1_100.loc[bond_comb_1_100.index >= "2009-08-12 00:00:00"]

bond_comb_1_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_TU_"+str(500) + ".pkl")
bond_comb_1_500.insert(0, "RVOL", ind_fcts_1_TU.values[(-bond_comb_1_500.shape[0]):, 0])
bond_comb_1_500.columns = comb_methods
bond_comb_1_500.index = ind_fcts_1_TU.index[(-bond_comb_1_500.shape[0]):]
bond_comb_1_500 = bond_comb_1_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_500))+1)]
bond_comb_1_500 = bond_comb_1_500.loc[bond_comb_1_500.index >= "2009-08-12 00:00:00"]

bond_comb_5_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_TU_"+str(100) + ".pkl")
bond_comb_5_100.insert(0, "RVOL", ind_fcts_5_TU.values[(-bond_comb_5_100.shape[0]):, 0])
bond_comb_5_100.columns = comb_methods
bond_comb_5_100.index = ind_fcts_5_TU.index[(-bond_comb_5_100.shape[0]):]
bond_comb_5_100 = bond_comb_5_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_100))+1)]
bond_comb_5_100 = bond_comb_5_100.loc[bond_comb_5_100.index >= "2009-08-12 00:00:00"]

bond_comb_5_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_TU_"+str(500) + ".pkl")
bond_comb_5_500.insert(0, "RVOL", ind_fcts_5_TU.values[(-bond_comb_5_500.shape[0]):, 0])
bond_comb_5_500.columns = comb_methods
bond_comb_5_500.index = ind_fcts_5_TU.index[(-bond_comb_5_500.shape[0]):]
bond_comb_5_500 = bond_comb_5_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_500))+1)]
bond_comb_5_500 = bond_comb_5_500.loc[bond_comb_5_500.index >= "2009-08-12 00:00:00"]

bond_comb_22_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_TU_"+str(100) + ".pkl")
bond_comb_22_100.insert(0, "RVOL", ind_fcts_22_TU.values[(-bond_comb_22_100.shape[0]):, 0])
bond_comb_22_100.columns = comb_methods
bond_comb_22_100.index = ind_fcts_22_TU.index[(-bond_comb_22_100.shape[0]):]
bond_comb_22_100 = bond_comb_22_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_100))+1)]
bond_comb_22_100 = bond_comb_22_100.loc[bond_comb_22_100.index >= "2009-08-12 00:00:00"]

bond_comb_22_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_TU_"+str(500) + ".pkl")
bond_comb_22_500.insert(0, "RVOL", ind_fcts_22_TU.values[(-bond_comb_22_500.shape[0]):, 0])
bond_comb_22_500.columns = comb_methods
bond_comb_22_500.index = ind_fcts_22_TU.index[(-bond_comb_22_500.shape[0]):]
bond_comb_22_500 = bond_comb_22_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_500))+1)]
bond_comb_22_500 = bond_comb_22_500.loc[bond_comb_22_500.index >= "2009-08-12 00:00:00"]


# figure
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
# cycles
col_cyc = cycler('color', ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#4daf4a'])
# 1_100
a = axes[0, 0]
a.set_prop_cycle(col_cyc)
bond_comb_1_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 100')
a.set_xlabel('Time')
a.set_ylabel('h = 1', size='large')
a.set_ylim([0.0001, 0.0026])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 1_500
a = axes[0, 1]
a.set_prop_cycle(col_cyc)
bond_comb_1_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 500')
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.0026])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_100
a = axes[1, 0]
a.set_prop_cycle(col_cyc)
bond_comb_5_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 5', size='large')
a.set_ylim([0.0001, 0.0026])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_500
a = axes[1, 1]
a.set_prop_cycle(col_cyc)
bond_comb_5_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.0026])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_100
a = axes[2, 0]
a.set_prop_cycle(col_cyc)
bond_comb_22_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 22', size='large')
a.set_ylim([0.0001, 0.0026])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_500
a = axes[2, 1]
a.set_prop_cycle(col_cyc)
bond_comb_22_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.0026])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# whole figure
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(fig_path + "bond_comb_TU.pdf", bbox_inches='tight')


# FV (5 Year)
# load accuracy tables
acc_table_1_100 = ft.create_acc_table(df=ind_fcts_1_FV, w=100,proc="multiple",df_name="ind_fcts_1_FV_"+str(100))
acc_table_1_500 = ft.create_acc_table(df=ind_fcts_1_FV, w=500,proc="multiple",df_name="ind_fcts_1_FV_"+str(500))

acc_table_5_100 = ft.create_acc_table(df=ind_fcts_5_FV, w=100,proc="multiple",df_name="ind_fcts_5_FV_"+str(100))
acc_table_5_500 = ft.create_acc_table(df=ind_fcts_5_FV, w=500,proc="multiple",df_name="ind_fcts_5_FV_"+str(500))

acc_table_22_100 = ft.create_acc_table(df=ind_fcts_22_FV, w=100,proc="multiple",df_name="ind_fcts_22_FV_"+str(100))
acc_table_22_500 = ft.create_acc_table(df=ind_fcts_22_FV, w=500,proc="multiple",df_name="ind_fcts_22_FV_"+str(500))

# prepare datasets for plotting
bond_comb_1_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_FV_"+str(100) + ".pkl")
bond_comb_1_100.insert(0, "RVOL", ind_fcts_1_FV.values[(-bond_comb_1_100.shape[0]):, 0])
bond_comb_1_100.columns = comb_methods
bond_comb_1_100.index = ind_fcts_1_FV.index[(-bond_comb_1_100.shape[0]):]
bond_comb_1_100 = bond_comb_1_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_100))+1)]
bond_comb_1_100 = bond_comb_1_100.loc[bond_comb_1_100.index >= "2009-08-12 00:00:00"]

bond_comb_1_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_FV_"+str(500) + ".pkl")
bond_comb_1_500.insert(0, "RVOL", ind_fcts_1_FV.values[(-bond_comb_1_500.shape[0]):, 0])
bond_comb_1_500.columns = comb_methods
bond_comb_1_500.index = ind_fcts_1_FV.index[(-bond_comb_1_500.shape[0]):]
bond_comb_1_500 = bond_comb_1_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_500))+1)]
bond_comb_1_500 = bond_comb_1_500.loc[bond_comb_1_500.index >= "2009-08-12 00:00:00"]

bond_comb_5_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_FV_"+str(100) + ".pkl")
bond_comb_5_100.insert(0, "RVOL", ind_fcts_5_FV.values[(-bond_comb_5_100.shape[0]):, 0])
bond_comb_5_100.columns = comb_methods
bond_comb_5_100.index = ind_fcts_5_FV.index[(-bond_comb_5_100.shape[0]):]
bond_comb_5_100 = bond_comb_5_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_100))+1)]
bond_comb_5_100 = bond_comb_5_100.loc[bond_comb_5_100.index >= "2009-08-12 00:00:00"]

bond_comb_5_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_FV_"+str(500) + ".pkl")
bond_comb_5_500.insert(0, "RVOL", ind_fcts_5_FV.values[(-bond_comb_5_500.shape[0]):, 0])
bond_comb_5_500.columns = comb_methods
bond_comb_5_500.index = ind_fcts_5_FV.index[(-bond_comb_5_500.shape[0]):]
bond_comb_5_500 = bond_comb_5_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_500))+1)]
bond_comb_5_500 = bond_comb_5_500.loc[bond_comb_5_500.index >= "2009-08-12 00:00:00"]

bond_comb_22_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_FV_"+str(100) + ".pkl")
bond_comb_22_100.insert(0, "RVOL", ind_fcts_22_FV.values[(-bond_comb_22_100.shape[0]):, 0])
bond_comb_22_100.columns = comb_methods
bond_comb_22_100.index = ind_fcts_22_FV.index[(-bond_comb_22_100.shape[0]):]
bond_comb_22_100 = bond_comb_22_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_100))+1)]
bond_comb_22_100 = bond_comb_22_100.loc[bond_comb_22_100.index >= "2009-08-12 00:00:00"]

bond_comb_22_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_FV_"+str(500) + ".pkl")
bond_comb_22_500.insert(0, "RVOL", ind_fcts_22_FV.values[(-bond_comb_22_500.shape[0]):, 0])
bond_comb_22_500.columns = comb_methods
bond_comb_22_500.index = ind_fcts_22_FV.index[(-bond_comb_22_500.shape[0]):]
bond_comb_22_500 = bond_comb_22_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_500))+1)]
bond_comb_22_500 = bond_comb_22_500.loc[bond_comb_22_500.index >= "2009-08-12 00:00:00"]


# figure
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
# cycles
col_cyc = cycler('color', ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#4daf4a'])
# 1_100
a = axes[0, 0]
a.set_prop_cycle(col_cyc)
bond_comb_1_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 100')
a.set_xlabel('Time')
a.set_ylabel('h = 1', size='large')
a.set_ylim([0.0001, 0.008])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 1_500
a = axes[0, 1]
a.set_prop_cycle(col_cyc)
bond_comb_1_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 500')
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.008])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_100
a = axes[1, 0]
a.set_prop_cycle(col_cyc)
bond_comb_5_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 5', size='large')
a.set_ylim([0.0001, 0.008])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_500
a = axes[1, 1]
a.set_prop_cycle(col_cyc)
bond_comb_5_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.008])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_100
a = axes[2, 0]
a.set_prop_cycle(col_cyc)
bond_comb_22_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 22', size='large')
a.set_ylim([0.0001, 0.008])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_500
a = axes[2, 1]
a.set_prop_cycle(col_cyc)
bond_comb_22_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.008])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# whole figure
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(fig_path + "bond_comb_FV.pdf", bbox_inches='tight')

# TY (10 Year)
# load accuracy tables
acc_table_1_100 = ft.create_acc_table(df=ind_fcts_1_TY, w=100,proc="multiple",df_name="ind_fcts_1_TY_"+str(100))
acc_table_1_500 = ft.create_acc_table(df=ind_fcts_1_TY, w=500,proc="multiple",df_name="ind_fcts_1_TY_"+str(500))

acc_table_5_100 = ft.create_acc_table(df=ind_fcts_5_TY, w=100,proc="multiple",df_name="ind_fcts_5_TY_"+str(100))
acc_table_5_500 = ft.create_acc_table(df=ind_fcts_5_TY, w=500,proc="multiple",df_name="ind_fcts_5_TY_"+str(500))

acc_table_22_100 = ft.create_acc_table(df=ind_fcts_22_TY, w=100,proc="multiple",df_name="ind_fcts_22_TY_"+str(100))
acc_table_22_500 = ft.create_acc_table(df=ind_fcts_22_TY, w=500,proc="multiple",df_name="ind_fcts_22_TY_"+str(500))

# prepare datasets for plotting
bond_comb_1_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_TY_"+str(100) + ".pkl")
bond_comb_1_100.insert(0, "RVOL", ind_fcts_1_TY.values[(-bond_comb_1_100.shape[0]):, 0])
bond_comb_1_100.columns = comb_methods
bond_comb_1_100.index = ind_fcts_1_TY.index[(-bond_comb_1_100.shape[0]):]
bond_comb_1_100 = bond_comb_1_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_100))+1)]
bond_comb_1_100 = bond_comb_1_100.loc[bond_comb_1_100.index >= "2009-08-12 00:00:00"]

bond_comb_1_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_TY_"+str(500) + ".pkl")
bond_comb_1_500.insert(0, "RVOL", ind_fcts_1_TY.values[(-bond_comb_1_500.shape[0]):, 0])
bond_comb_1_500.columns = comb_methods
bond_comb_1_500.index = ind_fcts_1_TY.index[(-bond_comb_1_500.shape[0]):]
bond_comb_1_500 = bond_comb_1_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_500))+1)]
bond_comb_1_500 = bond_comb_1_500.loc[bond_comb_1_500.index >= "2009-08-12 00:00:00"]

bond_comb_5_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_TY_"+str(100) + ".pkl")
bond_comb_5_100.insert(0, "RVOL", ind_fcts_5_TY.values[(-bond_comb_5_100.shape[0]):, 0])
bond_comb_5_100.columns = comb_methods
bond_comb_5_100.index = ind_fcts_5_TY.index[(-bond_comb_5_100.shape[0]):]
bond_comb_5_100 = bond_comb_5_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_100))+1)]
bond_comb_5_100 = bond_comb_5_100.loc[bond_comb_5_100.index >= "2009-08-12 00:00:00"]

bond_comb_5_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_TY_"+str(500) + ".pkl")
bond_comb_5_500.insert(0, "RVOL", ind_fcts_5_TY.values[(-bond_comb_5_500.shape[0]):, 0])
bond_comb_5_500.columns = comb_methods
bond_comb_5_500.index = ind_fcts_5_TY.index[(-bond_comb_5_500.shape[0]):]
bond_comb_5_500 = bond_comb_5_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_500))+1)]
bond_comb_5_500 = bond_comb_5_500.loc[bond_comb_5_500.index >= "2009-08-12 00:00:00"]

bond_comb_22_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_TY_"+str(100) + ".pkl")
bond_comb_22_100.insert(0, "RVOL", ind_fcts_22_TY.values[(-bond_comb_22_100.shape[0]):, 0])
bond_comb_22_100.columns = comb_methods
bond_comb_22_100.index = ind_fcts_22_TY.index[(-bond_comb_22_100.shape[0]):]
bond_comb_22_100 = bond_comb_22_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_100))+1)]
bond_comb_22_100 = bond_comb_22_100.loc[bond_comb_22_100.index >= "2009-08-12 00:00:00"]

bond_comb_22_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_TY_"+str(500) + ".pkl")
bond_comb_22_500.insert(0, "RVOL", ind_fcts_22_TY.values[(-bond_comb_22_500.shape[0]):, 0])
bond_comb_22_500.columns = comb_methods
bond_comb_22_500.index = ind_fcts_22_TY.index[(-bond_comb_22_500.shape[0]):]
bond_comb_22_500 = bond_comb_22_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_500))+1)]
bond_comb_22_500 = bond_comb_22_500.loc[bond_comb_22_500.index >= "2009-08-12 00:00:00"]


# figure
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
# cycles
col_cyc = cycler('color', ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#4daf4a'])
# 1_100
a = axes[0, 0]
a.set_prop_cycle(col_cyc)
bond_comb_1_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 100')
a.set_xlabel('Time')
a.set_ylabel('h = 1', size='large')
a.set_ylim([0.0001, 0.014])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 1_500
a = axes[0, 1]
a.set_prop_cycle(col_cyc)
bond_comb_1_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 500')
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.014])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_100
a = axes[1, 0]
a.set_prop_cycle(col_cyc)
bond_comb_5_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 5', size='large')
a.set_ylim([0.0001, 0.014])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_500
a = axes[1, 1]
a.set_prop_cycle(col_cyc)
bond_comb_5_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.014])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_100
a = axes[2, 0]
a.set_prop_cycle(col_cyc)
bond_comb_22_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 22', size='large')
a.set_ylim([0.0001, 0.014])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_500
a = axes[2, 1]
a.set_prop_cycle(col_cyc)
bond_comb_22_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0001, 0.014])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# whole figure
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(fig_path + "bond_comb_TY.pdf", bbox_inches='tight')

# US (30 Year)
# load accuracy tables
acc_table_1_100 = ft.create_acc_table(df=ind_fcts_1_US, w=100,proc="multiple",df_name="ind_fcts_1_US_"+str(100))
acc_table_1_500 = ft.create_acc_table(df=ind_fcts_1_US, w=500,proc="multiple",df_name="ind_fcts_1_US_"+str(500))

acc_table_5_100 = ft.create_acc_table(df=ind_fcts_5_US, w=100,proc="multiple",df_name="ind_fcts_5_US_"+str(100))
acc_table_5_500 = ft.create_acc_table(df=ind_fcts_5_US, w=500,proc="multiple",df_name="ind_fcts_5_US_"+str(500))

acc_table_22_100 = ft.create_acc_table(df=ind_fcts_22_US, w=100,proc="multiple",df_name="ind_fcts_22_US_"+str(100))
acc_table_22_500 = ft.create_acc_table(df=ind_fcts_22_US, w=500,proc="multiple",df_name="ind_fcts_22_US_"+str(500))

# prepare datasets for plotting
bond_comb_1_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_US_"+str(100) + ".pkl")
bond_comb_1_100.insert(0, "RVOL", ind_fcts_1_US.values[(-bond_comb_1_100.shape[0]):, 0])
bond_comb_1_100.columns = comb_methods
bond_comb_1_100.index = ind_fcts_1_US.index[(-bond_comb_1_100.shape[0]):]
bond_comb_1_100 = bond_comb_1_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_100))+1)]
bond_comb_1_100 = bond_comb_1_100.loc[bond_comb_1_100.index >= "2009-08-12 00:00:00"]

bond_comb_1_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(1)+"_US_"+str(500) + ".pkl")
bond_comb_1_500.insert(0, "RVOL", ind_fcts_1_US.values[(-bond_comb_1_500.shape[0]):, 0])
bond_comb_1_500.columns = comb_methods
bond_comb_1_500.index = ind_fcts_1_US.index[(-bond_comb_1_500.shape[0]):]
bond_comb_1_500 = bond_comb_1_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_1_500))+1)]
bond_comb_1_500 = bond_comb_1_500.loc[bond_comb_1_500.index >= "2009-08-12 00:00:00"]

bond_comb_5_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_US_"+str(100) + ".pkl")
bond_comb_5_100.insert(0, "RVOL", ind_fcts_5_US.values[(-bond_comb_5_100.shape[0]):, 0])
bond_comb_5_100.columns = comb_methods
bond_comb_5_100.index = ind_fcts_5_US.index[(-bond_comb_5_100.shape[0]):]
bond_comb_5_100 = bond_comb_5_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_100))+1)]
bond_comb_5_100 = bond_comb_5_100.loc[bond_comb_5_100.index >= "2009-08-12 00:00:00"]

bond_comb_5_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(5)+"_US_"+str(500) + ".pkl")
bond_comb_5_500.insert(0, "RVOL", ind_fcts_5_US.values[(-bond_comb_5_500.shape[0]):, 0])
bond_comb_5_500.columns = comb_methods
bond_comb_5_500.index = ind_fcts_5_US.index[(-bond_comb_5_500.shape[0]):]
bond_comb_5_500 = bond_comb_5_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_5_500))+1)]
bond_comb_5_500 = bond_comb_5_500.loc[bond_comb_5_500.index >= "2009-08-12 00:00:00"]

bond_comb_22_100 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_US_"+str(100) + ".pkl")
bond_comb_22_100.insert(0, "RVOL", ind_fcts_22_US.values[(-bond_comb_22_100.shape[0]):, 0])
bond_comb_22_100.columns = comb_methods
bond_comb_22_100.index = ind_fcts_22_US.index[(-bond_comb_22_100.shape[0]):]
bond_comb_22_100 = bond_comb_22_100.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_100))+1)]
bond_comb_22_100 = bond_comb_22_100.loc[bond_comb_22_100.index >= "2009-08-12 00:00:00"]

bond_comb_22_500 = pd.read_pickle(data_path + "Multiproc/MP_" + "ind_fcts_"+str(22)+"_US_"+str(500) + ".pkl")
bond_comb_22_500.insert(0, "RVOL", ind_fcts_22_US.values[(-bond_comb_22_500.shape[0]):, 0])
bond_comb_22_500.columns = comb_methods
bond_comb_22_500.index = ind_fcts_22_US.index[(-bond_comb_22_500.shape[0]):]
bond_comb_22_500 = bond_comb_22_500.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_22_500))+1)]
bond_comb_22_500 = bond_comb_22_500.loc[bond_comb_22_500.index >= "2009-08-12 00:00:00"]


# figure
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
# cycles
col_cyc = cycler('color', ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#4daf4a'])
# 1_100
a = axes[0, 0]
a.set_prop_cycle(col_cyc)
bond_comb_1_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 100')
a.set_xlabel('Time')
a.set_ylabel('h = 1', size='large')
a.set_ylim([0.0023, 0.021])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 1_500
a = axes[0, 1]
a.set_prop_cycle(col_cyc)
bond_comb_1_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_title('w = 500')
a.set_xlabel('Time')
a.set_ylim([0.0023, 0.021])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_100
a = axes[1, 0]
a.set_prop_cycle(col_cyc)
bond_comb_5_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 5', size='large')
a.set_ylim([0.0023, 0.021])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 5_500
a = axes[1, 1]
a.set_prop_cycle(col_cyc)
bond_comb_5_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0023, 0.021])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_100
a = axes[2, 0]
a.set_prop_cycle(col_cyc)
bond_comb_22_100.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylabel('h = 22', size='large')
a.set_ylim([0.0023, 0.021])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# 22_500
a = axes[2, 1]
a.set_prop_cycle(col_cyc)
bond_comb_22_500.plot(ax=a, linewidth=0.5, alpha=0.7)
a.set_xlabel('Time')
a.set_ylim([0.0023, 0.021])
a.grid(color='k', linestyle=':', linewidth=0.5)
a.legend(loc = 1, ncol=2, prop={'size': 8}, framealpha=1, fancybox=True)

# whole figure
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(fig_path + "bond_comb_US.pdf", bbox_inches='tight')

# END OF FILE
