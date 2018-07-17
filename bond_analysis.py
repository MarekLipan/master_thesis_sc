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
acc_table_ind_fcts_1_TU = ft.create_acc_table(df=ind_fcts_1_TU, w=40,
                                        proc="single",
                                        df_name="ind_fcts_1_TU")


# export accuracy tables to tex
ft.gen_tex_table(tbl=acc_table_ind_fcts_1_TU,
                 cap="Test",
                 file_name="ind_fcts_1_TU",
                 r=3)

# END OF FILE
