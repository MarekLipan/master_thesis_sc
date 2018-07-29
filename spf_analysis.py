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
from pylatex import Table, Tabular, MultiColumn, MultiRow, Tabu
from pylatex.utils import NoEscape
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler

# set the seed for replicability of results
random.seed(444)
np.random.seed(444)

# create accuracy tables
#acc_table_RGDP_1Y = ft.create_acc_table(df=spf_bal_RGDP_1Y, w=40,
#                                        proc="single",
#                                        df_name="spf_bal_RGDP_1Y")
# testing
#w=35
#for i in range(spf_bal_RGDP_1Y.shape[0]-w):
#    start_time = time.time()
#    df_train = spf_bal_RGDP_1Y.iloc[i:(w+i), :]
#    df_test = spf_bal_RGDP_1Y.iloc[(w+i):(w+i+1), 1:]
#    ############################
#    fcts = pd.concat([
#            cm.Equal_Weights(df_test),
#            cm.Bates_Granger_1(df_train, df_test),
#            cm.Bates_Granger_2(df_train, df_test),
#            cm.Bates_Granger_3(df_train, df_test, alpha=0.6),
#            cm.Bates_Granger_4(df_train, df_test, W=1.5),
#            cm.Bates_Granger_5(df_train, df_test, W=1.5),
#            cm.Granger_Ramanathan_1(df_train, df_test),
#            cm.Granger_Ramanathan_2(df_train, df_test),
#            cm.Granger_Ramanathan_3(df_train, df_test),
#            cm.AFTER(df_train, df_test, lambd=0.15),
#            cm.Median_Forecast(df_test),
#            cm.Trimmed_Mean_Forecast(df_test, alpha=0.05),
#            cm.PEW(df_train, df_test),
#            cm.Principal_Component_Forecast(df_train, df_test, "single"),
#            cm.Principal_Component_Forecast(df_train, df_test, "AIC"),
#            cm.Principal_Component_Forecast(df_train, df_test, "BIC"),
#            cm.Empirical_Bayes_Estimator(df_train, df_test),
#            cm.Kappa_Shrinkage(df_train, df_test, kappa=0.5),
#            cm.Two_Step_Egalitarian_LASSO(df_train, df_test, k_cv=5,
#                                          grid_l=-6, grid_h=2, grid_size=20),
#            #cm.Two_Step_Egalitarian_LASSO(df_train, df_test, k_cv=5,
#            #                              grid_l=-20, grid_h=2, grid_size=20),
#            cm.BMA_Marginal_Likelihood(df_train, df_test, iterations=6000,
#                                       burnin=1000, p_1=0.5),
#            cm.BMA_Predictive_Likelihood(df_train, df_test,
#                                         iterations=6000, burnin=1000,
#                                         p_1=0.5, l_share=0.7),
#            #cm.BMA_Marginal_Likelihood_exh(df_train, df_test),
#            #cm.BMA_Predictive_Likelihood_exh(df_train, df_test, l_share=0.1),
#            cm.ANN(df_train, df_test),
#            cm.EP_NN(df_train, df_test, sigma=0.05, gen=200, n=16),
#            cm.Bagging(df_train, df_test, B=500, threshold=1.96),
#            #cm.Bagging(df_train, df_test, B=500, threshold=1.28),
#            cm.Componentwise_Boosting(df_train, df_test, nu=0.1),
#            cm.AdaBoost(df_train, df_test, phi=0.1),
#            cm.cAPM_Constant(df_train, df_test, MaxRPT_r1=0.9, MaxRPT=0.01,
#                             no_rounds=10),
#            cm.cAPM_Q_learning(df_train, df_test, MinRPT=0.0001, MaxRPT_r1=0.9,
#                               MaxRPT=0.01, alpha=0.7, no_rounds=10),
#            cm.MK(df_train, df_test)
#    ], axis=1).values[0]
#    ############################
#    end_time = time.time()
##    print(str(i) + ": " + str(end_time-start_time))
#
#w=35
#for i in range(spf_bal_RGDP_1Y.shape[0]-w):
#    start_time = time.time()
#    df_train = spf_bal_RGDP_1Y.iloc[i:(w+i), :]
#    df_test = spf_bal_RGDP_1Y.iloc[(w+i):(w+i+1), 1:]
#    ############################
#    pred = cm.Two_Step_Egalitarian_LASSO(df_train, df_test, k_cv=5,
#                                  grid_l=-6, grid_h=2, grid_size=20)
#    ############################
#    end_time = time.time()
#    print(str(i) + ": " + str(end_time-start_time))
#    print("prediction: " + str(pred.values[0][0]))
#
#
#for w in [25, 35, 45]:
#
#    # load the data and compute accuracy measures
#    acc_table_RGDP_1Y = ft.create_acc_table(df=spf_bal_RGDP_1Y, w=w,
#                                            proc="multiple",
#                                            df_name="spf_bal_RGDP_1Y_"+str(w))
#
#    acc_table_RGDP_2Y = ft.create_acc_table(df=spf_bal_RGDP_2Y, w=w,
#                                           proc="multiple",
#                                            df_name="spf_bal_RGDP_2Y_"+str(w))
#
#    acc_table_HICP_1Y = ft.create_acc_table(df=spf_bal_HICP_1Y, w=w,
#                                            proc="multiple",
#                                            df_name="spf_bal_HICP_1Y_"+str(w))
#    acc_table_HICP_2Y = ft.create_acc_table(df=spf_bal_HICP_2Y, w=w,
#                                            proc="multiple",
#                                            df_name="spf_bal_HICP_2Y_"+str(w))
#   acc_table_UNEM_1Y = ft.create_acc_table(df=spf_bal_UNEM_1Y, w=w,
#                                            proc="multiple",
#                                            df_name="spf_bal_UNEM_1Y_"+str(w))
#    acc_table_UNEM_2Y = ft.create_acc_table(df=spf_bal_UNEM_2Y, w=w,
#                                            proc="multiple",
#                                            df_name="spf_bal_UNEM_2Y_"+str(w))
#
#    # export accuracy tables to tex
#    ft.gen_tex_table(tbl=acc_table_RGDP_1Y,
#                     cap="Real GDP - 1Y forecast horizon (w="+str(w)+")",
#                    file_name="spf_RGDP_1Y_"+str(w),
#                     r=3)
#
#    ft.gen_tex_table(tbl=acc_table_RGDP_2Y,
#                     cap="Real GDP - 2Y forecast horizon (w="+str(w)+")",
#                     file_name="spf_RGDP_2Y_"+str(w),
#                     r=3)
#
#    ft.gen_tex_table(tbl=acc_table_HICP_1Y,
#                     cap="HICP - 1Y forecast horizon (w="+str(w)+")",
#                     file_name="spf_HICP_1Y_"+str(w),
#                     r=3)
#
#    ft.gen_tex_table(tbl=acc_table_HICP_2Y,
#                     cap="HICP - 2Y forecast horizon (w="+str(w)+")",
#                     file_name="spf_HICP_2Y_"+str(w),
#                     r=3)
#
#    ft.gen_tex_table(tbl=acc_table_UNEM_1Y,
#                     cap="Unemployment - 1Y forecast horizon (w="+str(w)+")",
#                     file_name="spf_UNEM_1Y_"+str(w),
#                    r=3)
#
#    ft.gen_tex_table(tbl=acc_table_UNEM_2Y,
#                     cap="Unemployment - 2Y forecast horizon (w="+str(w)+")",
#                     file_name="spf_UNEM_2Y_"+str(w),
#                     r=3)

##########
# OUTPUTS#
##########
# path to where the figures and tables are stored
fig_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Figures/"
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"

# MULTITABLES
# load the data
for w in [25, 35, 45]:

    # load the data and compute accuracy measures
    acc_table_RGDP_1Y = ft.create_acc_table(df=spf_bal_RGDP_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_RGDP_1Y_"+str(w))

    acc_table_RGDP_2Y = ft.create_acc_table(df=spf_bal_RGDP_2Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_RGDP_2Y_"+str(w))

    acc_table_HICP_1Y = ft.create_acc_table(df=spf_bal_HICP_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_HICP_1Y_"+str(w))
    acc_table_HICP_2Y = ft.create_acc_table(df=spf_bal_HICP_2Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_HICP_2Y_"+str(w))
    acc_table_UNEM_1Y = ft.create_acc_table(df=spf_bal_UNEM_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_UNEM_1Y_"+str(w))
    acc_table_UNEM_2Y = ft.create_acc_table(df=spf_bal_UNEM_2Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_UNEM_2Y_"+str(w))

    # concatenate the tables together
    spf_multitable = pd.concat(
            [acc_table_RGDP_1Y, acc_table_RGDP_2Y,
             acc_table_HICP_1Y, acc_table_HICP_2Y,
             acc_table_UNEM_1Y, acc_table_UNEM_2Y],
             axis=1, join_axes=[acc_table_RGDP_1Y.index])

    # create table object
    tabl = Table()
    tabl.add_caption("Performance of forecast combinations of ECB SPF forecasts using the training window of the length: "+str(w))
    tabl.append(NoEscape('\label{tab: spf_comb_perf_'+str(w)+'}'))
    # create tabular object
    tabr = Tabular(table_spec="c|l" + 6*"|ccc")
    tabr.add_hline()
    tabr.add_hline()
    # header row
    tabr.add_row((MultiRow(3, data="Class"), MultiRow(3, data="Forecast Combination Method"),
                  MultiColumn(6, align='|c|', data="RGDP"),
                  MultiColumn(6, align='|c|', data="HICP"),
                  MultiColumn(6, align='|c', data="UNEM")))
    tabr.add_hline(start=3, end=20, cmidruleoption="lr")
    tabr.add_row(("", "",
                  MultiColumn(3, align='|c|', data="1Y"),
                  MultiColumn(3, align='|c|', data="2Y"),
                  MultiColumn(3, align='|c|', data="1Y"),
                  MultiColumn(3, align='|c|', data="2Y"),
                  MultiColumn(3, align='|c|', data="1Y"),
                  MultiColumn(3, align='|c|', data="2Y")))
    tabr.add_hline(start=3, end=20, cmidruleoption="lr")
    tabr.add_row(2*[""] + 6*["RMSE", "MAE", "MAPE"])
    tabr.add_hline()
    # fill in the rows of tabular
    # Simple
    tabr.add_row([MultiRow(13, data="Simple")] + [spf_multitable.index[0]] + [
            "{:.2f}".format(item) for item in spf_multitable.iloc[0, :]])
    for i in range(1, 13):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    tabr.add_hline()
    # Factor Analytic
    tabr.add_row([MultiRow(3, data="Factor An.")] + [spf_multitable.index[13]] + [
            "{:.2f}".format(item) for item in spf_multitable.iloc[13, :]])
    for i in range(14, 16):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    tabr.add_hline()
    # Shrinkage
    tabr.add_row([MultiRow(3, data="Shrinkage")] + [spf_multitable.index[16]] + [
            "{:.2f}".format(item) for item in spf_multitable.iloc[16, :]])
    for i in range(17, 19):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    tabr.add_hline()
    # BMA
    tabr.add_row([MultiRow(2, data="BMA")] + [spf_multitable.index[19]] + [
            "{:.2f}".format(item) for item in spf_multitable.iloc[19, :]])
    for i in range(20, 21):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    tabr.add_hline()
    # Alternative
    tabr.add_row([MultiRow(5, data="Alternative")] + [spf_multitable.index[21]] + [
            "{:.2f}".format(item) for item in spf_multitable.iloc[21, :]])
    for i in range(22, 26):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    tabr.add_hline()
    # APM
    tabr.add_row([MultiRow(3, data="APM")] + [spf_multitable.index[26]] + [
            "{:.2f}".format(item) for item in spf_multitable.iloc[26, :]])
    for i in range(27, 29):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    tabr.add_hline()
    for i in range(29, 32):
        tabr.add_row([""] + [spf_multitable.index[i]] + [
                "{:.2f}".format(item) for item in spf_multitable.iloc[i, :]])

    # end of table
    tabr.add_hline()
    # add tabular to table
    tabl.append(tabr)
    # export the table
    tabl.generate_tex(tab_path + "spf_comb_perf_"+str(w))


# FORECAST COMBINATION FIGURES
comb_methods = ['ECB', 'Equal Weights', 'Bates-Granger (1)', 'Bates-Granger (2)',
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

for w in [25, 35, 45]:
    # load accuracy tables
    acc_table_RGDP_1Y = ft.create_acc_table(df=spf_bal_RGDP_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_RGDP_1Y_"+str(w))

    acc_table_RGDP_2Y = ft.create_acc_table(df=spf_bal_RGDP_2Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_RGDP_2Y_"+str(w))

    acc_table_HICP_1Y = ft.create_acc_table(df=spf_bal_HICP_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_HICP_1Y_"+str(w))
    acc_table_HICP_2Y = ft.create_acc_table(df=spf_bal_HICP_2Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_HICP_2Y_"+str(w))
    acc_table_UNEM_1Y = ft.create_acc_table(df=spf_bal_UNEM_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_UNEM_1Y_"+str(w))
    acc_table_UNEM_2Y = ft.create_acc_table(df=spf_bal_UNEM_2Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_UNEM_2Y_"+str(w))
    # load all the data
    data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"
    # RGDP 1Y
    spf_comb_RGDP_1Y = pd.read_pickle(data_path + "Multiproc/MP_" + "spf_bal_RGDP_1Y_" + str(w) + ".pkl")
    spf_comb_RGDP_1Y.insert(0, "ECB", spf_bal_RGDP_1Y.values[(-spf_comb_RGDP_1Y.shape[0]):, 0])
    spf_comb_RGDP_1Y.columns = comb_methods
    spf_comb_RGDP_1Y.index = spf_bal_RGDP_1Y.index[(-spf_comb_RGDP_1Y.shape[0]):]
    spf_comb_RGDP_1Y = spf_comb_RGDP_1Y.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_RGDP_1Y))+1)]

    # RGDP 2Y
    spf_comb_RGDP_2Y = pd.read_pickle(data_path + "Multiproc/MP_" + "spf_bal_RGDP_2Y_" + str(w) + ".pkl")
    spf_comb_RGDP_2Y.insert(0, "ECB", spf_bal_RGDP_2Y.values[(-spf_comb_RGDP_2Y.shape[0]):, 0])
    spf_comb_RGDP_2Y.columns = comb_methods
    spf_comb_RGDP_2Y.index = spf_bal_RGDP_2Y.index[(-spf_comb_RGDP_2Y.shape[0]):]
    spf_comb_RGDP_2Y = spf_comb_RGDP_2Y.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_RGDP_2Y))+1)]

    # HICP 1Y
    spf_comb_HICP_1Y = pd.read_pickle(data_path + "Multiproc/MP_" + "spf_bal_HICP_1Y_" + str(w) + ".pkl")
    spf_comb_HICP_1Y.insert(0, "ECB", spf_bal_HICP_1Y.values[(-spf_comb_HICP_1Y.shape[0]):, 0])
    spf_comb_HICP_1Y.columns = comb_methods
    spf_comb_HICP_1Y.index = spf_bal_HICP_1Y.index[(-spf_comb_HICP_1Y.shape[0]):]
    spf_comb_HICP_1Y = spf_comb_HICP_1Y.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_HICP_1Y))+1)]

    # HICP 2Y
    spf_comb_HICP_2Y = pd.read_pickle(data_path + "Multiproc/MP_" + "spf_bal_HICP_2Y_" + str(w) + ".pkl")
    spf_comb_HICP_2Y.insert(0, "ECB", spf_bal_HICP_2Y.values[(-spf_comb_HICP_2Y.shape[0]):, 0])
    spf_comb_HICP_2Y.columns = comb_methods
    spf_comb_HICP_2Y.index = spf_bal_HICP_2Y.index[(-spf_comb_HICP_2Y.shape[0]):]
    spf_comb_HICP_2Y = spf_comb_HICP_2Y.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_HICP_2Y))+1)]

    # UNEM 1Y
    spf_comb_UNEM_1Y = pd.read_pickle(data_path + "Multiproc/MP_" + "spf_bal_UNEM_1Y_" + str(w) + ".pkl")
    spf_comb_UNEM_1Y.insert(0, "ECB", spf_bal_UNEM_1Y.values[(-spf_comb_UNEM_1Y.shape[0]):, 0])
    spf_comb_UNEM_1Y.columns = comb_methods
    spf_comb_UNEM_1Y.index = spf_bal_UNEM_1Y.index[(-spf_comb_UNEM_1Y.shape[0]):]
    spf_comb_UNEM_1Y = spf_comb_UNEM_1Y.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_UNEM_1Y))+1)]

    # UNEM 2Y
    spf_comb_UNEM_2Y = pd.read_pickle(data_path + "Multiproc/MP_" + "spf_bal_UNEM_2Y_" + str(w) + ".pkl")
    spf_comb_UNEM_2Y.insert(0, "ECB", spf_bal_UNEM_2Y.values[(-spf_comb_UNEM_2Y.shape[0]):, 0])
    spf_comb_UNEM_2Y.columns = comb_methods
    spf_comb_UNEM_2Y.index = spf_bal_UNEM_2Y.index[(-spf_comb_UNEM_2Y.shape[0]):]
    spf_comb_UNEM_2Y = spf_comb_UNEM_2Y.iloc[:, np.append(0, am.best_in_class(am.rank_methods(acc_table_UNEM_2Y))+1)]

    # figure
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
    # cycles
    col_cyc = cycler('color', ['#e41a1c', '#377eb8', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#4daf4a'])
    mar_cyc = cycler('marker', ['.']+5*['']+['.'])
    # RGDP (1Y)
    a = axes[0, 0]
    a.set_prop_cycle(col_cyc+mar_cyc)
    spf_comb_RGDP_1Y.plot(ax=a, linewidth=1, alpha=0.7)
    a.set_title('Real GDP Growth (1 Year Horizon)')
    a.set_xlabel('Time')
    a.grid(color='k', linestyle=':', linewidth=0.5)
    #a.set_ylim([-9, 6])
    a.legend(loc = 4, ncol=2, prop={'size': 7}, framealpha=1, fancybox=True)

    # RGDP (2Y)
    a = axes[0, 1]
    a.set_prop_cycle(col_cyc+mar_cyc)
    spf_comb_RGDP_2Y.plot(ax=a, linewidth=1, alpha=0.7)
    a.set_title('Real GDP Growth (2 Year Horizon)')
    a.set_xlabel('Time')
    a.grid(color='k', linestyle=':', linewidth=0.5)
    #a.set_ylim([-9, 6])
    a.legend(loc = 4, ncol=2, prop={'size': 7}, framealpha=1, fancybox=True)

    # HICP (1Y)
    a = axes[1, 0]
    a.set_prop_cycle(col_cyc+mar_cyc)
    spf_comb_HICP_1Y.plot(ax=a, linewidth=1, alpha=0.7)
    a.set_title('Harmonised Inflation (1 Year Horizon)')
    a.set_xlabel('Time')
    a.grid(color='k', linestyle=':', linewidth=0.5)
    a.legend(loc = 1, ncol=2, prop={'size': 7}, framealpha=1, fancybox=True)

    # HICP (2Y)
    a = axes[1, 1]
    a.set_prop_cycle(col_cyc+mar_cyc)
    spf_comb_HICP_2Y.plot(ax=a, linewidth=1, alpha=0.7)
    a.set_title('Harmonised Inflation (2 Year Horizon)')
    a.set_xlabel('Time')
    a.grid(color='k', linestyle=':', linewidth=0.5)
    a.legend(loc = 3, ncol=2, prop={'size': 7}, framealpha=1, fancybox=True)

    # UNEM (1Y)
    a = axes[2, 0]
    a.set_prop_cycle(col_cyc+mar_cyc)
    spf_comb_UNEM_1Y.plot(ax=a, linewidth=1, alpha=0.7)
    a.set_title('Unemployment Rate (1 Year Horizon)')
    a.set_xlabel('Time')
    a.grid(color='k', linestyle=':', linewidth=0.5)
    a.legend(loc = 4, ncol=2, prop={'size': 7}, framealpha=1, fancybox=True)

    # UNEM (2Y)
    a = axes[2, 1]
    a.set_prop_cycle(col_cyc+mar_cyc)
    spf_comb_UNEM_2Y.plot(ax=a, linewidth=1, alpha=0.7)
    a.set_title('Unemployment Rate (2 Year Horizon)')
    a.set_xlabel('Time')
    a.grid(color='k', linestyle=':', linewidth=0.5)
    a.legend(loc = 4, ncol=2, prop={'size': 7}, framealpha=1, fancybox=True)

    # whole figure
    fig.autofmt_xdate()
    fig.savefig(fig_path + "spf_comb_fcts_"+str(w)+".pdf", bbox_inches='tight')

#############
# RANK TABLE#
#############
acc_table = acc_table_RGDP_1Y
# END OF FILE
