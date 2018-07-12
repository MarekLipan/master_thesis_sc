"""
US Bond Volatility Forecasting

DATA PREPARATION SCRIPT

This script is used to download, clean a manipulate the data before analysis.

"""

import pandas as pd

# load the returns data
ret = pd.read_csv('/Users/Marek/Dropbox/Master_Thesis/Data/Bonds/US_ret.csv',
                  index_col="time_index")

# load the volatility data
rvol = pd.read_csv('/Users/Marek/Dropbox/Master_Thesis/Data/Bonds/US_rvol.csv',
                   index_col="time_index")


# index to datetime value
ret.index = pd.to_datetime(ret.index)
rvol.index = pd.to_datetime(rvol.index)

# sort by index
ret.sort_index(inplace=True)
rvol.sort_index(inplace=True)

# path to bond DataFrames
data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"
bond_data_path = data_path + "Bonds/"

# save the prepared DFs
ret.to_pickle(bond_data_path + "ret.pkl")
rvol.to_pickle(bond_data_path + "rvol.pkl")

###############
# END OF FILE #
###############
