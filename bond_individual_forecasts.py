"""
U.S. Government Bonds

BOND INDIVIDUAL FORECASTS SCRIPT

This script is used to estimate the individual models for realized volatility
forecasting

"""

import pandas as pd
import numpy as np

# set the seed for replicability of results
random.seed(444)
np.random.seed(444)

# ADF test null: unit root
adfuller(rvol.iloc[:, 0])
adfuller(rvol.iloc[:, 1])
adfuller(rvol.iloc[:, 2])
adfuller(rvol.iloc[:, 3])

#######################
# Individual forecasts#
#######################

# sample length
T = ret.shape[0]

# number of series
M = ret.shape[1]

# length of the rolling window
rw = 1000

# types of individual forecasts
ind_fcts_colnames = ["RV", "RiskMetrics", "HAR"]

# matrices of individual forecasts - 1 step ahead
ind_fcts_1_TU = pd.DataFrame(
        data=np.full((T-rw, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_1_FV = pd.DataFrame(
        data=np.full((T-rw, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_1_TY = pd.DataFrame(
        data=np.full((T-rw, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_1_US = pd.DataFrame(
        data=np.full((T-rw, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
# matrices of individual forecasts - 5 step ahead
ind_fcts_5_TU = pd.DataFrame(
        data=np.full((T-rw-4, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_5_FV = pd.DataFrame(
        data=np.full((T-rw-4, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_5_TY = pd.DataFrame(
        data=np.full((T-rw-4, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_5_US = pd.DataFrame(
        data=np.full((T-rw-4, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
# matrices of individual forecasts - 22 step ahead
ind_fcts_22_TU = pd.DataFrame(
        data=np.full((T-rw-21, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_22_FV = pd.DataFrame(
        data=np.full((T-rw-21, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_22_TY = pd.DataFrame(
        data=np.full((T-rw-21, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
        )
ind_fcts_22_US = pd.DataFrame(
        data=np.full((T-rw-21, len(ind_fcts_colnames)), 0, dtype=float),
        columns=ind_fcts_colnames
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

####################
# Historal Variance#
####################


##############
# RiskMetrics#
##############
fct_col = ind_fcts_colnames.index("RiskMetrics")

RiskMetrics = np.full((T, M), 0, dtype=float)

for t in range(1, T):
    
    RiskMetrics[t, :] = 0.94 * np.copy(RiskMetrics[t-1, :]) + 0.06 * (
            ret.iloc[t-1, :].values**2)

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


# END OF FILE
