"""
ECB Survey of Professional Forecasters

DATA PREPARATION SCRIPT

This script is used to download, clean a manipulate the data before analysis.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

# load the data
spf = pd.read_csv('/Users/Marek/Dropbox/Master_Thesis/Data/SPF/data.csv',
                  dtype={
                          "KEY": object,
                          "FREQ": object,
                          "REF_AREA": object,
                          "FCT_TOPIC": object,
                          "FCT_BREAKDOWN": object,
                          "FCT_HORIZON": object,
                          "SURVEY_FREQ": object,
                          "FCT_SOURCE": object,
                          "TIME_PERIOD": object,
                          "OBS_VALUE": float,
                          "OBS_STATUS": object,
                          "OBS_CONF": object,
                          "OBS_PRE_BREAK": float,
                          "OBS_COM": float,
                          "TIME_FORMAT": float,
                          "COLLECTION": object,
                          "COMPILING_ORG": float,
                          "DISS_ORG": float,
                          "DECIMALS": int,
                          "SOURCE_AGENCY": object,
                          "TITLE": object,
                          "TITLE_COMPL": object,
                          "UNIT": object,
                          "UNIT_MULT": int
                          }
                  )

# workaround for functioning caused by bug in jedi package
spf = pd.DataFrame(spf)

# global filter
spf = spf[
        (spf['FCT_TOPIC'].isin(["UNEM", "RGDP", "HICP"])) &
        (spf['FCT_BREAKDOWN'] == "POINT") &
        (~spf['FCT_SOURCE'].isin(["AVG", "VAR", "NUM"]))
        ]

###############################
# raw (unbalanced) panel data #
###############################


# function to create and fill a DataFrame with raw panel data
def create_raw_df(FCT_TOPIC, FREQ, FCT_HORIZON_prof,
                  FCT_HORIZON_ecb, t_start, t_end):

    # preparation DataFrame
    prep_df = pd.DataFrame(
                        spf[
                            (spf['FCT_TOPIC'] == FCT_TOPIC) &
                            (spf['FREQ'] == FREQ) &
                            (spf['FCT_HORIZON'].isin([FCT_HORIZON_prof,
                             FCT_HORIZON_ecb]))
                            ]
                        )

    # initialize the DataFrame
    df = pd.DataFrame(
                index=prep_df["TIME_PERIOD"].unique(),
                columns=prep_df['FCT_SOURCE'].unique()
                )

    # fill the DataFrame
    for i in df.index:
        for j in df.columns:
            input_value = prep_df[
                                    (prep_df["TIME_PERIOD"] == i) &
                                    (prep_df["FCT_SOURCE"] == j)
                                ].OBS_VALUE

            if input_value.empty:
                df.loc[i, j] = np.nan
            else:
                df.loc[i, j] = float(input_value)

    # index to datetime value
    df.index = pd.to_datetime(df.index)

    # sort by index
    df.sort_index(inplace=True)

    # cut the selected time period
    df = df[
            (df.index >= pd.to_datetime(t_start)) &
            (df.index <= pd.to_datetime(t_end))
           ]

    # return the raw DataFrame
    return pd.DataFrame(df)


# real GDP 1Y horizon
spf_raw_RGDP_1Y = create_raw_df(FCT_TOPIC="RGDP", FREQ="Q",
                                FCT_HORIZON_prof="P9M", FCT_HORIZON_ecb="M3M",
                                t_start="1999-07-01", t_end="2017-07-01")

# real GDP 2Y horizon
spf_raw_RGDP_2Y = create_raw_df(FCT_TOPIC="RGDP", FREQ="Q",
                                FCT_HORIZON_prof="P21M", FCT_HORIZON_ecb="M3M",
                                t_start="2000-07-01", t_end="2017-07-01")

# HICP (inflation) 1Y horizon
spf_raw_HICP_1Y = create_raw_df(FCT_TOPIC="HICP", FREQ="M",
                                FCT_HORIZON_prof="P12M", FCT_HORIZON_ecb="M0M",
                                t_start="1999-12-01", t_end="2017-12-01")

# HICP (inflation) 2Y horizon
spf_raw_HICP_2Y = create_raw_df(FCT_TOPIC="HICP", FREQ="M",
                                FCT_HORIZON_prof="P24M", FCT_HORIZON_ecb="M0M",
                                t_start="2000-12-01", t_end="2017-12-01")

# unemployment 1Y horizon, correction for delayed survey
spf_raw_UNEM_1Y = create_raw_df(FCT_TOPIC="UNEM", FREQ="M",
                                FCT_HORIZON_prof="P12M", FCT_HORIZON_ecb="M0M",
                                t_start="1999-11-01", t_end="2017-11-01")

spf_raw_UNEM_1Y.loc["2000-11-01", "001":] = np.array(
        spf_raw_UNEM_1Y.loc["2000-12-01", "001":]
        )

spf_raw_UNEM_1Y.loc["1999-11-01", "ECB"] = np.array(
        spf_raw_UNEM_1Y.loc["1999-12-01", "ECB"]
        )

spf_raw_UNEM_1Y = spf_raw_UNEM_1Y[~spf_raw_UNEM_1Y.index.isin(["2000-12-01",
                                                              "1999-12-01"])]


# unemployment 2Y horizon, correction for delayed survey
spf_raw_UNEM_2Y = create_raw_df(FCT_TOPIC="UNEM", FREQ="M",
                                FCT_HORIZON_prof="P24M", FCT_HORIZON_ecb="M0M",
                                t_start="2000-11-01", t_end="2017-11-01")

spf_raw_UNEM_2Y.loc["2001-11-01", "001":] = np.array(
        spf_raw_UNEM_2Y.loc["2001-12-01", "001":]
        )

spf_raw_UNEM_2Y = spf_raw_UNEM_2Y[spf_raw_UNEM_2Y.index != "2001-12-01"]

#######################
# balanced panel data #
#######################


# function to create and fill a DataFrame with balanced panel data
def create_balanced_df(df):
    """
    This function balances the panels of professional forecasters.

    For balancing, is used the method proposed by Genre et. al. (2013).
    Firstly, only the forecasters with no more than 4 consecutive observations
    are kept. Secondly, the missing observations are imputed using the linear
    filter. The beta is estimated from the following equation:

        Dev_{t} = Beta * Dev_{t-1} + Epsilon_{t},

    where Dev is deviation from the mean forecast of all forecasters. Then,
    the estimated beta is used to impute forward the missing observations and
    for simplification also missing observations from the beggining of the
    sample (in case there are any).


    Parameters
    ----------
    df : DataFrame
        DataFrame containing the raw dataset

    Returns
    -------
    DataFrame
        DataFrame containing the balanced panels

    """
    # initialize vector determining which collumns to keep
    keep = np.full(df.shape[1], True)

    # only professionals with no more than 4 consecutive NaNs are kept
    for j in range(df.shape[1]):
        # amount of consecutive NaNs of a given professional
        current_cons_nan = 0
        max_cons_nan = 0

        for i in range(df.shape[0]):

            if np.isnan(df.iloc[i, j]):
                current_cons_nan += 1

                # in case of the last row, update immediately
                if (i == df.shape[0]-1) & (current_cons_nan > max_cons_nan):
                    max_cons_nan = current_cons_nan
            else:
                if current_cons_nan > max_cons_nan:
                    max_cons_nan = current_cons_nan

                current_cons_nan = 0

        if max_cons_nan > 4:
            keep[j] = False

    # keep the frequently responding professionals
    df = pd.DataFrame(df.loc[:, keep])

    # for each professional (exclude ECB), calculate the deviation from the
    # mean forecast and regress it on its lagged value, then use the beta to
    # imput the missing values
    mean_forecast = np.mean(df.iloc[:, 1:], axis=1)

    for j in range(1, df.shape[1]):

        dev_from_mean = df.iloc[:, j] - mean_forecast
        lag_dev_from_mean = dev_from_mean.shift(periods=1)

        # drop the NaNs before running the regression
        reg_mat = pd.concat([lag_dev_from_mean, dev_from_mean],
                            axis=1).dropna()

        # create linear regression object
        lin_reg = linear_model.LinearRegression(fit_intercept=False)

        # fit the model
        lin_reg.fit(reg_mat.iloc[:, 0].values.reshape(-1, 1),
                    reg_mat.iloc[:, 1].values.reshape(-1, 1))

        # store the estimated beta
        beta = lin_reg.coef_

        # find the postition of first non-nan observation
        first_obs_pos = np.nan
        k = 0
        while (np.isnan(first_obs_pos) & (k <= df.shape[0])):
            if (~np.isnan(df.iloc[k, j])):
                first_obs_pos = k
            k += 1

        # forward imputation of missing observations
        for i in range(first_obs_pos+1, df.shape[0]):
            if (np.isnan(df.iloc[i, j])):
                # imputed value = mean forecast + beta * past deviation
                df.iloc[i, j] = float(
                        mean_forecast[i] + beta * dev_from_mean[i-1])
                # update the deviation from the mean vector
                dev_from_mean[i] = float(df.iloc[i, j] - mean_forecast[i])

        # backward imputation of missing observations
        if first_obs_pos != 0:
            for i in range(first_obs_pos-1, -1, -1):
                # mean forecast + beta fraction of the first measured
                # deviation (simplification)
                df.iloc[i, j] = float(
                        mean_forecast[i] + beta*dev_from_mean[first_obs_pos])

    return pd.DataFrame(df)


# real GDP 1Y horizon
spf_bal_RGDP_1Y = create_balanced_df(spf_raw_RGDP_1Y)

# real GDP 2Y horizon
spf_bal_RGDP_2Y = create_balanced_df(spf_raw_RGDP_2Y)

# HICP (inflation) 1Y horizon
spf_bal_HICP_1Y = create_balanced_df(spf_raw_HICP_1Y)

# HICP (inflation) 2Y horizon
spf_bal_HICP_2Y = create_balanced_df(spf_raw_HICP_2Y)

# unemployment 1Y horizon, correction for delayed survey
spf_bal_UNEM_1Y = create_balanced_df(spf_raw_UNEM_1Y)

# unemployment 2Y horizon, correction for delayed survey
spf_bal_UNEM_2Y = create_balanced_df(spf_raw_UNEM_2Y)

###############
# END OF FILE #
###############
