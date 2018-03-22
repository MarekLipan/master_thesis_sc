"""
ECB Survey of Professional Forecasters

ANALYSIS SCRIPT

This script is used to perform the main analysis, i.e. train and test models.

"""

import pandas as pd
import numpy as np
import combination_methods as cm

# define length of windows
train_win_len = 20
test_win_len = 5

# example train and test sets
df_train = spf_bal_RGDP_1Y.iloc[0:train_win_len, :]
df_test = spf_bal_RGDP_1Y.iloc[train_win_len:(train_win_len+test_win_len), 1:]

# test analysis
pred_1 = cm.Equal_weights(df_test)






# END OF FILE
