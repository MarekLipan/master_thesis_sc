"""
MAIN SCRIPT

This script is used to run the individual scripts.

"""

import pandas as pd
import numpy as np

# global paths
data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"

######################################
# Survey of Professional Forecasters #
######################################

# prepare the SPF DataFrames from the original file
#
# runfile('C:/Users/Marek/Desktop/IES/master_thesis_sc/spf_data_preparation.py',
#        wdir='C:/Users/Marek/Desktop/IES/master_thesis_sc')

# path to spf balanced DataFrames
spf_bal_data_path = data_path + "SPF/Balanced_panels/"

# save the prepared balanced DFs
#
# spf_bal_RGDP_1Y.to_pickle(spf_bal_data_path + "spf_bal_RGDP_1Y.pkl")
# spf_bal_RGDP_2Y.to_pickle(spf_bal_data_path + "spf_bal_RGDP_2Y.pkl")
# spf_bal_HICP_1Y.to_pickle(spf_bal_data_path + "spf_bal_HICP_1Y.pkl")
# spf_bal_HICP_2Y.to_pickle(spf_bal_data_path + "spf_bal_HICP_2Y.pkl")
# spf_bal_UNEM_1Y.to_pickle(spf_bal_data_path + "spf_bal_UNEM_1Y.pkl")
# spf_bal_UNEM_2Y.to_pickle(spf_bal_data_path + "spf_bal_UNEM_2Y.pkl")

# read the prepared balanced DFs
spf_bal_RGDP_1Y = pd.read_pickle(spf_bal_data_path + "spf_bal_RGDP_1Y.pkl")
spf_bal_RGDP_2Y = pd.read_pickle(spf_bal_data_path + "spf_bal_RGDP_2Y.pkl")
spf_bal_HICP_1Y = pd.read_pickle(spf_bal_data_path + "spf_bal_HICP_1Y.pkl")
spf_bal_HICP_2Y = pd.read_pickle(spf_bal_data_path + "spf_bal_HICP_2Y.pkl")
spf_bal_UNEM_1Y = pd.read_pickle(spf_bal_data_path + "spf_bal_UNEM_1Y.pkl")
spf_bal_UNEM_2Y = pd.read_pickle(spf_bal_data_path + "spf_bal_UNEM_2Y.pkl")


# analysis
# runfile('C:/Users/Marek/Desktop/IES/master_thesis_sc/spf_analysis.py',
#        wdir='C:/Users/Marek/Desktop/IES/master_thesis_sc')

# dataframes for development of new combining methods
df_train = spf_bal_RGDP_1Y.iloc[:40, :]
df_test = spf_bal_RGDP_1Y.iloc[40:44, 1:]

df = spf_bal_RGDP_1Y
w= 40

# parameters for development of new combining methods
iterations = 60000
burnin = 10000
p_1 = 0.5

# END OF FILE
