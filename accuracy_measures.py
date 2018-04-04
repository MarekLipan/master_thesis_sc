# -*- coding: utf-8 -*-
"""
ACCURACY MEASURES module

This module contains definitions of measures of forecast accuracy, which can
be used in the accuracy tables.

"""

import numpy as np

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


# THE END OF MODULE
