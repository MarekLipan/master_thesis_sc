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
#acc_table_RGDP_1Y = ft.create_acc_table(df=spf_bal_RGDP_1Y, w=40,
#                                        proc="single",
#                                        df_name="spf_bal_RGDP_1Y")

for w in [25, 35, 45]:

    acc_table_RGDP_1Y = ft.create_acc_table(df=spf_bal_RGDP_1Y, w=w,
                                            proc="multiple",
                                            df_name="spf_bal_RGDP_1Y_"+str(w))

    acc_table_RGDP_2Y = ft.create_acc_table(df=spf_bal_RGDP_2Y, w=40,
                                            proc="multiple",
                                            df_name="spf_bal_RGDP_2Y")
    acc_table_HICP_1Y = ft.create_acc_table(df=spf_bal_HICP_1Y, w=40,
                                            proc="multiple",
                                            df_name="spf_bal_HICP_1Y")
    acc_table_HICP_2Y = ft.create_acc_table(df=spf_bal_HICP_2Y, w=40,
                                            proc="multiple",
                                            df_name="spf_bal_HICP_2Y")
    acc_table_UNEM_1Y = ft.create_acc_table(df=spf_bal_UNEM_1Y, w=40,
                                            proc="multiple",
                                            df_name="spf_bal_UNEM_1Y")
    acc_table_UNEM_2Y = ft.create_acc_table(df=spf_bal_UNEM_2Y, w=40,
                                            proc="multiple",
                                            df_name="spf_bal_UNEM_2Y")

for w in [25, 35, 45]:
    # export accuracy tables to tex
    ft.gen_tex_table(tbl=acc_table_RGDP_1Y,
                     cap="Real GDP - 1Y forecast horizon (w="+str(w)+")",
                     file_name="spf_RGDP_1Y",
                     r=3)

    ft.gen_tex_table(tbl=acc_table_RGDP_2Y,
                     cap="Real GDP - 2Y forecast horizon",
                     file_name="spf_RGDP_2Y",
                     r=3)

    ft.gen_tex_table(tbl=acc_table_HICP_1Y,
                     cap="HICP - 1Y forecast horizon",
                     file_name="spf_HICP_1Y",
                     r=3)

    ft.gen_tex_table(tbl=acc_table_HICP_2Y,
                     cap="HICP - 2Y forecast horizon",
                     file_name="spf_HICP_2Y",
                     r=3)

    ft.gen_tex_table(tbl=acc_table_UNEM_1Y,
                     cap="Unemployment - 1Y forecast horizon",
                     file_name="spf_UNEM_1Y",
                     r=3)

    ft.gen_tex_table(tbl=acc_table_UNEM_2Y,
                     cap="Unemployment - 2Y forecast horizon",
                     file_name="spf_UNEM_2Y",
                     r=3)

# END OF FILE
