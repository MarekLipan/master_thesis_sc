# -*- coding: utf-8 -*-
"""
MULTIPROCESSING SCRIPT

This script is used for Ad-hoc fast parallel computations. This file must be
executed in the external system terminal.
"""

__spec__ = None

import multiprocessing as mp
import pandas as pd
import forecast_tables as ft
import numpy as np
import random
import time

# data path
path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"

# datasets to be loaded
df_names = ["spf_bal_RGDP_1Y", "spf_bal_RGDP_2Y", "spf_bal_HICP_1Y",
            "spf_bal_HICP_2Y", "spf_bal_UNEM_1Y", "spf_bal_UNEM_2Y"]

# load the datasets
df_list = []
df_len = []

for i in df_names:

    complete_path = path + "SPF/Balanced_panels/" + i + ".pkl"
    df_part = pd.read_pickle(complete_path)
    df_list.append(df_part)
    df_len.append(df_part.shape[0])

# length of the rolling window
w = 40


def roll_window(w_len, df_ind, obs_ind):
    """
    The function accepts an index, which defines the rolling window. The
    rolling out-of-sample forecasts are produced using the combination methods.

    Parameters
    ----------
    w_len : Float
        length of the rolling window

    df_ind : Float
        index of the DF in the list of DFs

    obs_ind : Float
        index of the observation in the given DF

    Returns
    -------
    DataFrame
        Forecasts produced using the combination methods.
    """

    # get parameters from the supplied tuple
    # w_len, df_ind, obs_ind = par[0], par[1], par[2]

    # create the datasets
    df = df_list[df_ind]
    df_train = df.iloc[obs_ind:(w_len+obs_ind), :]
    df_test = df.iloc[(w_len+obs_ind):(w_len+obs_ind+1), 1:]

    # set the seed for replicability of results
    # each process has fixed unique seed
    seed = w_len + df_ind + obs_ind
    random.seed(seed)
    np.random.seed(seed)

    # combine the forecasts
    ind_fcts = ft.run_comb_methods(df_train, df_test)

    return ind_fcts


if __name__ == '__main__':

    start_time = time.time()

    # list of tuples to be supplied to the rolling_window function
    # order : window length, dataset, observation
    par_list = []

    for e in range(len(df_list)):
        for f in range(df_len[e]-w):
            par_list.append((w, e, f))

    # build one large forecast table containing forecasts for all datasets
    p = mp.Pool()
    fcts_table = p.starmap(roll_window, par_list)
    p.close()
    p.join()

    # convert list of numpy arrays into a dataframe
    fcts_table_df = pd.DataFrame(fcts_table)

    # split the forecast table and save it
    fcts_table_arr = np.asarray(fcts_table)
    fcts_table_ind = np.delete(np.cumsum(np.asarray(df_len)-w), -1)
    fcts_table_split = np.split(fcts_table_arr, fcts_table_ind)

    for g in range(len(df_names)):
        save_path = path + "Multiproc/MP_" + df_names[g] + ".pkl"
        fcts_table_df = pd.DataFrame(fcts_table_split[g])
        fcts_table_df.to_pickle(save_path)

    end_time = time.time()
    print(end_time-start_time)
