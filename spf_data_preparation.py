"""
ECB Survey of Professional Forecasters

DATA PREPARATION SCRIPT

This script is used to download, clean a manipulate the data before analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

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
