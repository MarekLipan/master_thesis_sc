"""
ECB Survey of Professional Forecasters

ANALYSIS SCRIPT

This script is used to perform the main analysis, i.e. train and test models.

"""

import numpy as np
import forecast_tables as ft
import random

# set the seed for replicability of results
random.seed(444)
np.random.seed(444)

# create accuracy tables
acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=500,
                                        proc="single",
                                        df_name="ind_fcts_1_TU")


# all tables
acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=200,
                                        proc="multiple",
                                        df_name="ind_fcts_1_TU")

acc_table_ind_fcts_5_TU = ft.create_acc_table(df=ind_fcts_5_TU, w=200,
                                        proc="multiple",
                                        df_name="ind_fcts_5_TU")

acc_table_ind_fcts_22_TU = ft.create_acc_table(df=ind_fcts_22_TU, w=200,
                                        proc="multiple",
                                        df_name="ind_fcts_22_TU")


# export accuracy tables to tex
ft.gen_tex_table(tbl=acc_table_ind_fcts_1_TU,
                 cap="Combined 1-step-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures",
                 file_name="ind_fcts_1_TU",
                 r=6)

ft.gen_tex_table(tbl=acc_table_ind_fcts_5_TU,
                 cap="Combined 5-steps-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures",
                 file_name="ind_fcts_5_TU",
                 r=6)

ft.gen_tex_table(tbl=acc_table_ind_fcts_22_TU,
                 cap="Combined 22-steps-ahead forecasts of the realized volatility of log-returns of TU (2 Year) futures",
                 file_name="ind_fcts_22_TU",
                 r=6)

# END OF FILE
