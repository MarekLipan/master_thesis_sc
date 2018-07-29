# -*- coding: utf-8 -*-
"""
ACCURACY MEASURES module

This module contains definitions of measures of forecast accuracy, which can
be used in the accuracy tables.

"""

import numpy as np
from scipy.stats import rankdata

###########################
# Scale-dependent measures#
###########################


def RMSE(errors):
    """
    The function comutes Root Mean Square Error (RMSE) for a given vector
    of errors.

    Parameters
    ----------
    errors : NumpyArray
        Vector of forecast errors.

    Returns
    -------
    RMSE : numpy.float64
        Root Mean Square Error
    """

    RMSE = np.sqrt(np.mean(errors**2))

    return RMSE


def MAE(errors):
    """
    The function comutes Mean Absolute Error (MAE) for a given vector of
    errors.

    Parameters
    ----------
    errors : NumpyArray
        Vector of forecast errors.

    Returns
    -------
    MAE : numpy.float64
        Mean Absolute Error
    """

    MAE = np.mean(np.absolute(errors))

    return MAE

#####################################
# Measures based on percentage error#
#####################################


def MAPE(errors, y):
    """
    The function comutes Mean Absolute Percentage Error (MAPE) for a given
    vector of errors.

    Note: Observations equal to zero are disregarded, becasue division by zero
    breaks down the computation.

    Parameters
    ----------
    errors : NumpyArray
        Vector of forecast errors.

    y : NumpyArray
        Vector of realized (true) values.

    Returns
    -------
    MAPE : numpy.float64
        Mean Absolute Percentage Error
    """

    errors = errors[np.where(y != 0)]
    y = y[np.where(y != 0)]

    percentage_errors = 100 * (errors / y)

    MAPE = np.mean(np.absolute(percentage_errors))

    return MAPE

#######################################
# Functions using rankings in measures#
#######################################


def rank_methods(acc_table):
    """
    The function comutes the ranks for all combination methods according
    to each measure and then computes the average rank across measures.

    Parameters
    ----------
    acc_table : DataFrame
        Accuracy table, output from the "create_acc_table" function

    Returns
    -------
    rank_vec : NumpyArray
        Vector of average ranks for each method
    """
    RMSE_vec = acc_table.values[:(-3), 0]
    MAE_vec = acc_table.values[:(-3), 1]
    MAPE_vec = acc_table.values[:(-3), 2]
    rank_vec = (rankdata(RMSE_vec) + rankdata(MAE_vec) + rankdata(MAPE_vec))/3

    return rank_vec


def best_in_class(rank_vec):
    """
    The function selects the best best method in each class and outputs the
    list of indices of these methods

    Parameters
    ----------
    rank_vec : NumpyArray
        Output from the "rank_methods" function

    Returns
    -------
    best_vec : NumpyArray
        Vector of indices of best method from each forecast combination class
    """

    best_list = np.array([
            np.argmin(rank_vec[:13])+0,
            np.argmin(rank_vec[13:16])+13,
            np.argmin(rank_vec[16:19])+16,
            np.argmin(rank_vec[19:21])+19,
            np.argmin(rank_vec[21:26])+21,
            np.argmin(rank_vec[26:29])+26
            ], dtype=int)

    return best_list

# THE END OF MODULE
