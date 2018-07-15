"""
U.S. Government Bonds

BOND INDIVIDUAL FORECASTS SCRIPT

This script is used to estimate the individual models for realized volatility
forecasting

"""

import pandas as pd
import numpy as np
import itertools
import random
from statsmodels.tsa.api import VAR, AR

# set the seed for replicability of results
random.seed(444)
np.random.seed(444)

# ADF test null: unit root
#adfuller(rvol.iloc[:, 0])
#adfuller(rvol.iloc[:, 1])
#adfuller(rvol.iloc[:, 2])
#adfuller(rvol.iloc[:, 3])

#######################
# Individual forecasts#
#######################

# sample length
T = ret.shape[0]

# number of series
M = ret.shape[1]

# length of the rolling window
rw = 1000

# types of data
bond_data_colnames = ["TU", "FV", "TY", "US"]
# column names for VAR model individual forecasts
# combinations of variables
var_combs = []
var_combs_let = []
for i in range(1, 5):
    var_combs += list(itertools.combinations([0, 1, 2, 3], i))
    var_combs_let += list(itertools.combinations(bond_data_colnames, i))
# transform into lists
for i in range(len(var_combs)):
    var_combs[i] = list(var_combs[i])
    var_combs_let[i] = list(var_combs_let[i])
# connect VAR names
for i in range(len(var_combs_let)):
    var_combs_let

TU_VAR_colnames = []
FV_VAR_colnames = []
TY_VAR_colnames = []
US_VAR_colnames = []
for i in range(len(var_combs)):
    if 0 in var_combs[i]:
        TU_VAR_colnames.append("VAR ("+', '.join(var_combs_let[i])+")")
    if 1 in var_combs[i]:
        FV_VAR_colnames.append("VAR ("+', '.join(var_combs_let[i])+")")
    if 2 in var_combs[i]:
        TY_VAR_colnames.append("VAR ("+', '.join(var_combs_let[i])+")")
    if 3 in var_combs[i]:
        US_VAR_colnames.append("VAR ("+', '.join(var_combs_let[i])+")")

# types of individual forecasts
ind_fcts_colnames = ["RV", "Historical Volatility", "RiskMetrics", "HAR",
                     "GARCH"]
TU_colnames = ind_fcts_colnames + TU_VAR_colnames
FV_colnames = ind_fcts_colnames + FV_VAR_colnames
TY_colnames = ind_fcts_colnames + TY_VAR_colnames
US_colnames = ind_fcts_colnames + US_VAR_colnames
    
# matrices of individual forecasts - 1 step ahead
ind_fcts_1_TU = pd.DataFrame(
        data=np.full((T-rw, len(TU_colnames)), 0, dtype=float),
        columns=TU_colnames
        )
ind_fcts_1_FV = pd.DataFrame(
        data=np.full((T-rw, len(FV_colnames)), 0, dtype=float),
        columns=FV_colnames
        )
ind_fcts_1_TY = pd.DataFrame(
        data=np.full((T-rw, len(TY_colnames)), 0, dtype=float),
        columns=TY_colnames
        )
ind_fcts_1_US = pd.DataFrame(
        data=np.full((T-rw, len(US_colnames)), 0, dtype=float),
        columns=US_colnames
        )
# matrices of individual forecasts - 5 step ahead
ind_fcts_5_TU = pd.DataFrame(
        data=np.full((T-rw-4, len(TU_colnames)), 0, dtype=float),
        columns=TU_colnames
        )
ind_fcts_5_FV = pd.DataFrame(
        data=np.full((T-rw-4, len(FV_colnames)), 0, dtype=float),
        columns=FV_colnames
        )
ind_fcts_5_TY = pd.DataFrame(
        data=np.full((T-rw-4, len(TY_colnames)), 0, dtype=float),
        columns=TY_colnames
        )
ind_fcts_5_US = pd.DataFrame(
        data=np.full((T-rw-4, len(US_colnames)), 0, dtype=float),
        columns=US_colnames
        )
# matrices of individual forecasts - 22 step ahead
ind_fcts_22_TU = pd.DataFrame(
        data=np.full((T-rw-21, len(TU_colnames)), 0, dtype=float),
        columns=TU_colnames
        )
ind_fcts_22_FV = pd.DataFrame(
        data=np.full((T-rw-21, len(FV_colnames)), 0, dtype=float),
        columns=FV_colnames
        )
ind_fcts_22_TY = pd.DataFrame(
        data=np.full((T-rw-21, len(TY_colnames)), 0, dtype=float),
        columns=TY_colnames
        )
ind_fcts_22_US = pd.DataFrame(
        data=np.full((T-rw-21, len(US_colnames)), 0, dtype=float),
        columns=US_colnames
        )

#######################################
# Realized Volatiliy (RV, true values)#
#######################################
fct_col = ind_fcts_colnames.index("RV")

# TU
ind_fcts_1_TU.iloc[:, fct_col] = rvol.iloc[rw:, 0].values
ind_fcts_5_TU.iloc[:, fct_col] = rvol.iloc[rw+4:, 0].values
ind_fcts_22_TU.iloc[:, fct_col] = rvol.iloc[rw+21:, 0].values
# FV
ind_fcts_1_FV.iloc[:, fct_col] = rvol.iloc[rw:, 1].values
ind_fcts_5_FV.iloc[:, fct_col] = rvol.iloc[rw+4:, 1].values
ind_fcts_22_FV.iloc[:, fct_col] = rvol.iloc[rw+21:, 1].values
# TY
ind_fcts_1_TY.iloc[:, fct_col] = rvol.iloc[rw:, 2].values
ind_fcts_5_TY.iloc[:, fct_col] = rvol.iloc[rw+4:, 2].values
ind_fcts_22_TY.iloc[:, fct_col] = rvol.iloc[rw+21:, 2].values
# US
ind_fcts_1_US.iloc[:, fct_col] = rvol.iloc[rw:, 3].values
ind_fcts_5_US.iloc[:, fct_col] = rvol.iloc[rw+4:, 3].values
ind_fcts_22_US.iloc[:, fct_col] = rvol.iloc[rw+21:, 3].values

######################
# Historal Volatility#
######################
fct_col = ind_fcts_colnames.index("Historical Volatility")

# roll window trough the returns
for t in range(rw, T):
    hist_vol = np.std(ret.iloc[(t-rw):t, :], ddof=1)

    # save forecasts
    # 1 step ahead
    ind_fcts_1_TU.iloc[t-rw, fct_col] = hist_vol[0]
    ind_fcts_1_FV.iloc[t-rw, fct_col] = hist_vol[1]
    ind_fcts_1_TY.iloc[t-rw, fct_col] = hist_vol[2]
    ind_fcts_1_US.iloc[t-rw, fct_col] = hist_vol[3]
    if (t-rw) < (T-rw-4):  # 5 step ahead
        ind_fcts_5_TU.iloc[t-rw, fct_col] = hist_vol[0]
        ind_fcts_5_FV.iloc[t-rw, fct_col] = hist_vol[1]
        ind_fcts_5_TY.iloc[t-rw, fct_col] = hist_vol[2]
        ind_fcts_5_US.iloc[t-rw, fct_col] = hist_vol[3]
    if (t-rw) < (T-rw-21):  # 22 step ahead
        ind_fcts_22_TU.iloc[t-rw, fct_col] = hist_vol[0]
        ind_fcts_22_FV.iloc[t-rw, fct_col] = hist_vol[1]
        ind_fcts_22_TY.iloc[t-rw, fct_col] = hist_vol[2]
        ind_fcts_22_US.iloc[t-rw, fct_col] = hist_vol[3]

##############
# RiskMetrics#
##############
fct_col = ind_fcts_colnames.index("RiskMetrics")

RiskMetrics = np.full((T, M), 0, dtype=float)

for t in range(1, T):

    RiskMetrics[t, :] = 0.94 * np.copy(RiskMetrics[t-1, :]) + 0.06 * (
            np.sqrt(ret.iloc[t-1, :].values**2))

# out-of-sample forecasting
# TU
ind_fcts_1_TU.iloc[:, fct_col] = RiskMetrics[rw:, 0]
ind_fcts_5_TU.iloc[:, fct_col] = RiskMetrics[rw:-4, 0]
ind_fcts_22_TU.iloc[:, fct_col] = RiskMetrics[rw:-21:, 0]
# FV
ind_fcts_1_FV.iloc[:, fct_col] = RiskMetrics[rw:, 1]
ind_fcts_5_FV.iloc[:, fct_col] = RiskMetrics[rw:-4, 1]
ind_fcts_22_FV.iloc[:, fct_col] = RiskMetrics[rw:-21:, 1]
# TY
ind_fcts_1_TY.iloc[:, fct_col] = RiskMetrics[rw:, 2]
ind_fcts_5_TY.iloc[:, fct_col] = RiskMetrics[rw:-4, 2]
ind_fcts_22_TY.iloc[:, fct_col] = RiskMetrics[rw:-21:, 2]
# US
ind_fcts_1_US.iloc[:, fct_col] = RiskMetrics[rw:, 3]
ind_fcts_5_US.iloc[:, fct_col] = RiskMetrics[rw:-4, 3]
ind_fcts_22_US.iloc[:, fct_col] = RiskMetrics[rw:-21:, 3]

######
# HAR#
######
fct_col = ind_fcts_colnames.index("HAR")


def RV_average(series, ind, lag):
    output = 0
    for i in range(lag):
        output += series[ind-i-1]/lag
    return output


for m in range(M):
    # prepare the data
    HAR_data = np.full((T, 4), 0, dtype=float)
    for t in range(22, T):
        # original series
        HAR_data[t, 0] = rvol.iloc[t, m]
        # daily
        HAR_data[t, 1] = RV_average(rvol.iloc[:, m], t, 1)
        # weekly
        HAR_data[t, 2] = RV_average(rvol.iloc[:, m], t, 5)
        # monthly
        HAR_data[t, 3] = RV_average(rvol.iloc[:, m], t, 22)

    # estimate and forecast RV in a rolling window
    for t in range(rw, T):
        # select the training sample, first 22 observations are cut due to RV_m
        X = np.full((rw-22, 4), 1, dtype=float)
        X[:, 1:] = HAR_data[(t-rw+22):t, 1:]
        X_t = np.transpose(X)
        y = HAR_data[(t-rw+22):t, 0]
        # estimate OLS on the training sample
        beta_hat = np.linalg.multi_dot([np.linalg.inv(np.dot(X_t, X)), X_t, y])
        beta_const = beta_hat[0]
        beta_coeff = beta_hat[1:]
        # out-of-sample forecasting
        # compute forecasts iteratively
        RV_rolling_vec = y[-22:]
        for j in range(22):
            current_X = np.array([
                    # daily
                    RV_average(RV_rolling_vec, 22, 1),
                    # weekly
                    RV_average(RV_rolling_vec, 22, 5),
                    # monthly
                    RV_average(RV_rolling_vec, 22, 22)
                    ])
            # forecast
            current_fct = beta_const + np.dot(current_X, beta_coeff)

            # save forecasts
            if j == 0:  # 1 step ahead
                if m == 0:  # TU
                    ind_fcts_1_TU.iloc[t-rw, fct_col] = current_fct
                if m == 1:  # FV
                    ind_fcts_1_FV.iloc[t-rw, fct_col] = current_fct
                if m == 2:  # TY
                    ind_fcts_1_TY.iloc[t-rw, fct_col] = current_fct
                if m == 3:  # US
                    ind_fcts_1_US.iloc[t-rw, fct_col] = current_fct
            if j == 4 and (t-rw) < (T-rw-4):  # 5 step ahead
                if m == 0:  # TU
                    ind_fcts_5_TU.iloc[t-rw, fct_col] = current_fct
                if m == 1:  # FV
                    ind_fcts_5_FV.iloc[t-rw, fct_col] = current_fct
                if m == 2:  # TY
                    ind_fcts_5_TY.iloc[t-rw, fct_col] = current_fct
                if m == 3:  # US
                    ind_fcts_5_US.iloc[t-rw, fct_col] = current_fct
            if j == 21 and (t-rw) < (T-rw-21):  # 22 step ahead
                if m == 0:  # TU
                    ind_fcts_22_TU.iloc[t-rw, fct_col] = current_fct
                if m == 1:  # FV
                    ind_fcts_22_FV.iloc[t-rw, fct_col] = current_fct
                if m == 2:  # TY
                    ind_fcts_22_TY.iloc[t-rw, fct_col] = current_fct
                if m == 3:  # US
                    ind_fcts_22_US.iloc[t-rw, fct_col] = current_fct

        # roll the RV vector 1 observation forward
        RV_rolling_vec = np.append(RV_rolling_vec[1:], current_fct)

########
# GARCH#
########
fct_col = ind_fcts_colnames.index("GARCH")

# import data from R (estimation using rugarch is prefered over
# the current implementation arch python library)
GARCH_1 = pd.read_csv(
        '/Users/Marek/Dropbox/Master_Thesis/Data/Bonds/GARCH_1.csv'
        )
GARCH_5 = pd.read_csv(
        '/Users/Marek/Dropbox/Master_Thesis/Data/Bonds/GARCH_5.csv'
        )
GARCH_22 = pd.read_csv(
        '/Users/Marek/Dropbox/Master_Thesis/Data/Bonds/GARCH_22.csv'
        )
# save the forecasts
ind_fcts_1_TU.iloc[:, fct_col] = GARCH_1.iloc[:, 0].values
ind_fcts_5_TU.iloc[:, fct_col] = GARCH_5.iloc[:, 0].values
ind_fcts_22_TU.iloc[:, fct_col] = GARCH_22.iloc[:, 0].values
# FV
ind_fcts_1_FV.iloc[:, fct_col] = GARCH_1.iloc[:, 1].values
ind_fcts_5_FV.iloc[:, fct_col] = GARCH_5.iloc[:, 1].values
ind_fcts_22_FV.iloc[:, fct_col] = GARCH_22.iloc[:, 1].values
# TY
ind_fcts_1_TY.iloc[:, fct_col] = GARCH_1.iloc[:, 2].values
ind_fcts_5_TY.iloc[:, fct_col] = GARCH_5.iloc[:, 2].values
ind_fcts_22_TY.iloc[:, fct_col] = GARCH_22.iloc[:, 2].values
# US
ind_fcts_1_US.iloc[:, fct_col] = GARCH_1.iloc[:, 3].values
ind_fcts_5_US.iloc[:, fct_col] = GARCH_5.iloc[:, 3].values
ind_fcts_22_US.iloc[:, fct_col] = GARCH_22.iloc[:, 3].values

######
# VAR#
######
# fractional differencing parameter estimated using GPH method in R
# fractionally differencing series
# def frac_diff(series, d):
#    new series ...
#    output = 0
#    for i in range(1, 101):
#        denum = np.math.factorial(i)
#        num = 1
#        for j in range(i):
#            num *= (d-j)
#        output += (series[ind - i] * (num/denum))
#    return output

# roll trough the training data
for c in var_combs:
    # translate into variable names
    c_vars = "VAR ("+', '.join([bond_data_colnames[x] for x in c])+")"
    # select variables to VAR
    X_all = rvol.iloc[:, c].values
    # select time window
    for t in range(rw, T):
        X = X_all[(t-rw):t, :]
        # for one variable run simple AR
        if len(c) == 1:
            # estimate the model
            model = AR(X)
            model_fit = model.fit(5)
            model_fct = model_fit.predict(start=rw, end=(rw+21))
            # save forecasts
            if 0 in c:  # check for TU
                ind_fcts_1_TU.iloc[(t-rw), TU_colnames.index(c_vars)] = model_fct[0]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_TU.iloc[(t-rw), TU_colnames.index(c_vars)] = model_fct[4]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_TU.iloc[(t-rw), TU_colnames.index(c_vars)] = model_fct[21]
            if 1 in c:  # check for FV
                ind_fcts_1_FV.iloc[(t-rw), FV_colnames.index(c_vars)] = model_fct[0]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_FV.iloc[(t-rw), FV_colnames.index(c_vars)] = model_fct[4]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_FV.iloc[(t-rw), FV_colnames.index(c_vars)] = model_fct[21]
            if 2 in c:  # check for TY
                ind_fcts_1_TY.iloc[(t-rw), TY_colnames.index(c_vars)] = model_fct[0]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_TY.iloc[(t-rw), TY_colnames.index(c_vars)] = model_fct[4]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_TY.iloc[(t-rw), TY_colnames.index(c_vars)] = model_fct[21]
            if 3 in c:  # check for US
                ind_fcts_1_US.iloc[(t-rw), US_colnames.index(c_vars)] = model_fct[0]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_US.iloc[(t-rw), US_colnames.index(c_vars)] = model_fct[4]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_US.iloc[(t-rw), US_colnames.index(c_vars)] = model_fct[21]
        # for more than one variable run VAR
        else:
            # estimate the model
            model = VAR(X)
            model_fit = model.fit(5)
            model_fct = model_fit.forecast(X, 22)
            # save forecasts
            if 0 in c:  # check for TU
                ind_fcts_1_TU.iloc[(t-rw), TU_colnames.index(c_vars)] = model_fct[0, c.index(0)]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_TU.iloc[(t-rw), TU_colnames.index(c_vars)] = model_fct[4, c.index(0)]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_TU.iloc[(t-rw), TU_colnames.index(c_vars)] = model_fct[21, c.index(0)]
            if 1 in c:  # check for FV
                ind_fcts_1_FV.iloc[(t-rw), FV_colnames.index(c_vars)] = model_fct[0, c.index(1)]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_FV.iloc[(t-rw), FV_colnames.index(c_vars)] = model_fct[4, c.index(1)]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_FV.iloc[(t-rw), FV_colnames.index(c_vars)] = model_fct[21, c.index(1)]
            if 2 in c:  # check for TY
                ind_fcts_1_TY.iloc[(t-rw), TY_colnames.index(c_vars)] = model_fct[0, c.index(2)]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_TY.iloc[(t-rw), TY_colnames.index(c_vars)] = model_fct[4, c.index(2)]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_TY.iloc[(t-rw), TY_colnames.index(c_vars)] = model_fct[21, c.index(2)]
            if 3 in c:  # check for US
                ind_fcts_1_US.iloc[(t-rw), US_colnames.index(c_vars)] = model_fct[0, c.index(3)]
                if (t-rw) < (T-rw-4):
                    ind_fcts_5_US.iloc[(t-rw), US_colnames.index(c_vars)] = model_fct[4, c.index(3)]
                if (t-rw) < (T-rw-21):
                    ind_fcts_22_US.iloc[(t-rw), US_colnames.index(c_vars)] = model_fct[21, c.index(3)]
        

# END OF FILE
