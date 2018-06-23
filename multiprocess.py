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

# load the data
data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"
spf_bal_data_path = data_path + "SPF/Balanced_panels/"
df = pd.read_pickle(spf_bal_data_path + "spf_bal_RGDP_1Y.pkl")

# length of the rolling window
w = 40


def window_40(ind):
    """
    The function accepts an index, which defines the rolling window. The
    rolling out-of-sample forecasts are produced using the combination methods.

    Parameters
    ----------
    ind : Float
        index

    Returns
    -------
    DataFrame
        Forecasts produced using the combination methods.
    """

    # create the datasets
    df_train = df.iloc[ind:(40+ind), :]
    df_test = df.iloc[(40+ind):(40+ind+1), 1:]  # must be of type DataFrame

    # set the seed for replicability of results
    # each process has fixed unique seed
    random.seed(ind)
    np.random.seed(ind)

    # combine the forecasts
    ind_fcts = ft.run_comb_methods(df_train, df_test)

    return ind_fcts


if __name__ == '__main__':

    # indices pointing on the out-of-sample observation after the rolling win.
    T = df.shape[0]
    oos_T = T - w  # no. one-step-ahead out of sample forecasts

    p = mp.Pool()
    fcts_table = p.map(window_40, range(oos_T))
    p.close()
    p.join()

    # convert list of numpy arrays into a dataframe
    fcts_table_df = pd.DataFrame(fcts_table)

    # save the forecasts table
    fcts_table_df.to_pickle(data_path + "Multiproc/MP_data.pkl")
