"""
ECB Survey of Professional Forecasters

ANALYSIS SCRIPT

This script is used to perform the main analysis, i.e. train and test models.

"""

import pandas as pd
import numpy as np
import forecast_tables as ft

# create accuracy tables
acc_table_RGDP_1Y = ft.create_acc_table(
        df=spf_bal_RGDP_1Y, w=40)

# export accuracy tables to tex
ft.gen_tex_table(tbl=acc_table_RGDP_1Y,
                 cap="Real GDP - 1Y forecast horizon",
                 file_name="spf_RGDP_1Y",
                 r=3)

# END OF FILE
