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

# set the seed for replicability of results
random.seed(444)
np.random.seed(444)


# testing
w=100
for i in range(ind_fcts_1_TU.shape[0]-w):
    start_time = time.time()
    df_train = ind_fcts_1_TU.iloc[i:(w+i), :]
    df_test = ind_fcts_1_TU.iloc[(w+i):(w+i+1), 1:]
    ############################
    fcts = pd.concat([
        cm.Equal_Weights(df_test),
        cm.Bates_Granger_1(df_train, df_test),
        cm.Bates_Granger_2(df_train, df_test),
        cm.Bates_Granger_3(df_train, df_test, alpha=0.6),
        cm.Bates_Granger_4(df_train, df_test, W=1.5),
        cm.Bates_Granger_5(df_train, df_test, W=1.5),
        cm.Granger_Ramanathan_1(df_train, df_test),
        cm.Granger_Ramanathan_2(df_train, df_test),
        cm.Granger_Ramanathan_3(df_train, df_test),
        cm.AFTER(df_train, df_test, lambd=0.15),
        cm.Median_Forecast(df_test),
        cm.Trimmed_Mean_Forecast(df_test, alpha=0.05),
        cm.PEW(df_train, df_test),
        cm.Principal_Component_Forecast(df_train, df_test, "single"),
        cm.Principal_Component_Forecast(df_train, df_test, "AIC"),
        cm.Principal_Component_Forecast(df_train, df_test, "BIC"),
        cm.Empirical_Bayes_Estimator(df_train, df_test),
        cm.Kappa_Shrinkage(df_train, df_test, kappa=0.5),
        cm.Two_Step_Egalitarian_LASSO(df_train, df_test, k_cv=5,
                                      grid_l=-20, grid_h=2, grid_size=20),
        cm.BMA_Marginal_Likelihood_exh(df_train, df_test),
        cm.BMA_Predictive_Likelihood_exh(df_train, df_test, l_share=0.1),
        cm.ANN(df_train, df_test),
        cm.EP_NN(df_train, df_test, sigma=0.05, gen=200, n=16),
        cm.Bagging(df_train, df_test, B=500, threshold=1.28),
        cm.Componentwise_Boosting(df_train, df_test, nu=0.1),
        cm.AdaBoost(df_train, df_test, phi=0.1),
        cm.cAPM_Constant(df_train, df_test, MaxRPT_r1=0.9, MaxRPT=0.01,
                         no_rounds=10),
        cm.cAPM_Q_learning(df_train, df_test, MinRPT=0.0001, MaxRPT_r1=0.9,
                           MaxRPT=0.01, alpha=0.7, no_rounds=10),
        cm.MK(df_train, df_test)
    ], axis=1).values[0]
    ############################
    end_time = time.time()
    print(str(i) + ": " + str(end_time-start_time))



for i in range(ind_fcts_1_TU.shape[0]-w):
    start_time = time.time()
    df_train = ind_fcts_1_TU.iloc[i:(w+i), :]
    df_test = ind_fcts_1_TU.iloc[(w+i):(w+i+1), 1:]
    ############################
    pred = cm.BMA_Predictive_Likelihood_exh(df_train, df_test, l_share=0.1)
    ############################
    end_time = time.time()
    print(str(i) + ": " + str(end_time-start_time))
    print("prediction: " + str(pred.values[0][0]))


# create accuracy tables
acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=500,
                                        proc="single",
                                        df_name="ind_fcts_1_TU_"+str(w))


# all tables
acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=500,
                                        proc="multiple",
                                        df_name="ind_fcts_1_TU_"+str(w))

acc_table_ind_fcts_5_TU = ft.create_acc_table(df=ind_fcts_5_TU, w=500,
                                        proc="multiple",
                                        df_name="ind_fcts_5_TU_"+str(w))

acc_table_ind_fcts_22_TU = ft.create_acc_table(df=ind_fcts_22_TU, w=500,
                                        proc="multiple",
                                        df_name="ind_fcts_22_TU_"+str(w))


# export accuracy tables to tex
ft.gen_tex_table(tbl=acc_table_ind_fcts_1_TU,
                 cap="Combined 1-step-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures",
                 file_name="ind_fcts_1_TU",
                 r=6)

ft.gen_tex_table(tbl=acc_table_ind_fcts_5_TU,
                 cap="Combined 5-steps-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures",
                 file_name="ind_fcts_5_TU",
                 r=6)

ft.gen_tex_table(tbl=acc_table_ind_fcts_22_TU,
                 cap="Combined 22-steps-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures",
                 file_name="ind_fcts_22_TU",
                 r=6)

# END OF FILE
