"""
MAIN SCRIPT

This script is used to run the individual scripts.

"""

import pandas as pd

# switchers
run_spf = 0
run_bond = 1

# global paths
data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"

######################################
# Survey of Professional Forecasters #
######################################
if run_spf == 1:
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

    # testing dataframes for development of new combining methods
    df_train = spf_bal_RGDP_1Y.iloc[:40, :]
    df_test = spf_bal_RGDP_1Y.iloc[40:44, 1:]

    df = spf_bal_RGDP_1Y
    w = 40

    # parameters for development of new combining methods

##################################
# US Bond Volatility Forecasting #
##################################
if run_bond == 1:
    # prepare the US Bond DataFrames from the original file
    # runfile('C:/Users/Marek/Desktop/IES/master_thesis_sc/bond_data_preparation.py',
    #         wdir='C:/Users/Marek/Desktop/IES/master_thesis_sc')

    # path to bond DataFrames
    bond_data_path = data_path + "Bonds/"

    # read the prepared DFs
    ret = pd.read_pickle(bond_data_path + "ret.pkl")
    rvol = pd.read_pickle(bond_data_path + "rvol.pkl")
    
    # read prepared individual forecasts
    ind_fcts_1_TU = pd.read_pickle(bond_data_path + "ind_fcts_1_TU.pkl")
    ind_fcts_5_TU = pd.read_pickle(bond_data_path + "ind_fcts_5_TU.pkl")
    ind_fcts_22_TU = pd.read_pickle(bond_data_path + "ind_fcts_22_TU.pkl")
    ind_fcts_1_FV = pd.read_pickle(bond_data_path + "ind_fcts_1_FV.pkl")
    ind_fcts_5_FV = pd.read_pickle(bond_data_path + "ind_fcts_5_FV.pkl")
    ind_fcts_22_FV = pd.read_pickle(bond_data_path + "ind_fcts_22_FV.pkl")
    ind_fcts_1_TY = pd.read_pickle(bond_data_path + "ind_fcts_1_TY.pkl")
    ind_fcts_5_TY = pd.read_pickle(bond_data_path + "ind_fcts_5_TY.pkl")
    ind_fcts_22_TY = pd.read_pickle(bond_data_path + "ind_fcts_22_TY.pkl")
    ind_fcts_1_US = pd.read_pickle(bond_data_path + "ind_fcts_1_US.pkl")
    ind_fcts_5_US = pd.read_pickle(bond_data_path + "ind_fcts_5_US.pkl")
    ind_fcts_22_US = pd.read_pickle(bond_data_path + "ind_fcts_22_US.pkl")
    
    # testing dataframes for development of new combining methods
    df_train = ind_fcts_1_TU.iloc[:1000, :]
    df_test = ind_fcts_1_TU.iloc[1000:1001, 1:]

    df = ind_fcts_1_TU
    w = 200
    # parameters for development of new combining methods
    k_cv = 5

###############
# END OF FILE #
###############
