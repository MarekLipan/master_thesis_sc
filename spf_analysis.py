"""
ECB Survey of Professional Forecasters

ANALYSIS SCRIPT

This script is used to perform the main analysis, i.e. train and test models.

"""

import pandas as pd
import numpy as np
import combination_methods as cm

# define length of windows
train_win_len = 30
test_win_len = 10

# example train and test sets
df_train = spf_bal_RGDP_1Y.iloc[0:train_win_len, :]
df_test = spf_bal_RGDP_1Y.iloc[train_win_len:(train_win_len+test_win_len), 1:]

# parameters
nu = 25
alpha = 0.6
W = 1.5


# test analysis
pred1 = cm.Equal_weights(df_test)
pred2 = cm.Bates_Granger_1(df_train, df_test, nu=nu)
pred3 = cm.Bates_Granger_2(df_train, df_test, nu=nu)
pred4 = cm.Bates_Granger_3(df_train, df_test, nu=nu, alpha=alpha)
pred5 = cm.Bates_Granger_4(df_train, df_test, W=W)
pred6 = cm.Bates_Granger_5(df_train, df_test, W=W)
    
df_pred = pd.DataFrame({"Equal Weights": np.mean(df_test, axis=1)})

# END OF FILE
