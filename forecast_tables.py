# -*- coding: utf-8 -*-
"""
FORECAST TABLES module

This module contains definitions of functions dedicated to create and export
tables for the master thesis.

"""

import pandas as pd
import numpy as np
import combination_methods as cm
import accuracy_measures as am
from pylatex import Table, Tabular


def create_acc_table(df, w):
    """
    The function creates table of forecasts accuracy for basic forecast
    comparison.

    The used measures of accuracy are Root Mean Square Error (RMSE), Mean
    Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).


    Parameters
    ----------
    df : DataFrame
        DataFrame containing the  realized values in the first column and the
        individual forecasts in the other columns

    w : Integer
        Integer indicating the size of the training window.


    Returns
    -------
    String
        String for creating the basic forecast accuracy table in latex
    """

    # forecast combination methods
    comb_m = np.array(["Equal Weights",
                       "Bates-Granger (1)",
                       "Bates-Granger (2)",
                       "Bates-Granger (3)",
                       "Bates-Granger (4)",
                       "Bates-Granger (5)",
                       "Granger-Ramanathan (1)",
                       "Granger-Ramanathan (2)",
                       "Granger-Ramanathan (3)",
                       "AFTER",
                       "Median Forecast",
                       "Trimmed Mean Forecast",
                       "PEW",
                       "Principal Component Forecast",
                       "Principal Component Forecast (AIC)",
                       "Principal Component Forecast (BIC)",
                       "Empirical Bayes Estimator",
                       "Kappa-Shrinkage",
                       "Two-Step Egalitarian LASSO"
                       ])

    # define dimensions
    C = comb_m.shape[0]
    T = df.shape[0]
    oos_T = T - w  # no. one-step-ahead out of sample forecasts
    K = df.shape[1] - 1  # number of individual forecasts

    # accuracy measures
    measures = np.array(["RMSE", "MAE", "MAPE"])
    M = measures.size

    # initialize table to fill in forecasts, accuracy measures, realized values
    fcts_table = pd.DataFrame(data=np.full((C, oos_T), 0, dtype=float),
                              index=comb_m)
    acc_table = pd.DataFrame(data=np.full((C, M), 0, dtype=float),
                             columns=measures,
                             index=comb_m)
    real_val = np.full(oos_T, 0, dtype=float)

    # compute and store one-step-ahead forecasting
    for i in range(oos_T):

        # define training and testing sets
        df_train = df.iloc[i:(w+i), :]
        df_test = df.iloc[(w+i):(w+i+1), 1:]  # must be of type DataFrame

        # save realized (true) value
        real_val[i] = df.iloc[(w+i), 0]

        # combine and forecast
        fcts_table.iloc[:, i] = pd.concat([
                cm.Equal_Weights(df_test),
                cm.Bates_Granger_1(df_train, df_test, nu=40),
                cm.Bates_Granger_2(df_train, df_test, nu=40),
                cm.Bates_Granger_3(df_train, df_test, nu=40, alpha=0.6),
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
                cm.Two_Step_Egalitarian_LASSO(df_train, df_test, k_cv=5)
                ], axis=1).values[0]

    # compute and store accuracy measures for the combined forecasts
    for i in range(C):

        errors = real_val - fcts_table.iloc[i, :].values

        acc_table.iloc[i, :] = np.array([
                am.RMSE(errors),
                am.MAE(errors),
                am.MAPE(errors, real_val)
                ])

    # add best and worsts individual metrics
    ind_errors = np.array(df.iloc[w:, 1:].subtract(df.iloc[w:, 0], axis=0))
    ind_index = np.array(["Best Individual",
                          "Median Individual",
                          "Worst Individual"])
    ind_acc_table = pd.DataFrame(
            data=np.full((ind_index.size, M), np.nan, dtype=float),
            columns=measures,
            index=ind_index
            )

    # initialize vector to store calculated measures for individual forecasts
    ind_measure = np.full(K, np.nan, dtype=float)

    # RMSE
    for i in range(K):
        ind_measure[i] = am.RMSE(ind_errors[:, i])

    ind_acc_table.loc["Best Individual", "RMSE"] = min(ind_measure)
    ind_acc_table.loc["Median Individual", "RMSE"] = np.median(ind_measure)
    ind_acc_table.loc["Worst Individual", "RMSE"] = max(ind_measure)

    # MAE
    for i in range(K):
        ind_measure[i] = am.MAE(ind_errors[:, i])

    ind_acc_table.loc["Best Individual", "MAE"] = min(ind_measure)
    ind_acc_table.loc["Median Individual", "MAE"] = np.median(ind_measure)
    ind_acc_table.loc["Worst Individual", "MAE"] = max(ind_measure)

    # MAPE
    for i in range(K):
        ind_measure[i] = am.MAPE(ind_errors[:, i], real_val)

    ind_acc_table.loc["Best Individual", "MAPE"] = min(ind_measure)
    ind_acc_table.loc["Median Individual", "MAPE"] = np.median(ind_measure)
    ind_acc_table.loc["Worst Individual", "MAPE"] = max(ind_measure)

    # complete the accuracy table
    acc_table = acc_table.append(ind_acc_table)

    return acc_table


def gen_tex_table(tbl, cap, file_name, r):
    """
    The function creates a tex file to which it export the given table from
    python.

    Parameters
    ----------
    tbl : DataFrame
        DataFrame containing the table to be exported.

    cap : Str
        Table caption.

    file_name : Str
        Name of the tex file in the "Tables" directory.


    r : Int
       Number of decimals to round up the metrics.
    """

    # create tabule object
    tabl = Table()
    tabl.add_caption(cap)

    # create tabular object
    tabr = Tabular(table_spec="lccc")
    tabr.add_hline()
    tabr.add_hline()

    # header row
    tabr.add_row(["Forecast Combination Method"] + list(tbl))
    tabr.add_hline()

    # number of combination methods + additional rows
    R = tbl.shape[0]

    # fill in the rows for each combination method (-3 for individuals)
    for i in range(R-3):

        tabr.add_row([tbl.index[i]] + list(np.around(tbl.iloc[i, :],
                     decimals=r)))

    tabr.add_hline()

    # additional rows
    for i in range(R-3, R):

        tabr.add_row([tbl.index[i]] + list(np.around(tbl.iloc[i, :],
                     decimals=r)))

    # end of table
    tabr.add_hline()
    tabr.add_hline()

    # add tabular to table
    tabl.append(tabr)

    # export the table
    tabl.generate_tex("C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/" +
                      file_name)

    return

# THE END OF MODULE
