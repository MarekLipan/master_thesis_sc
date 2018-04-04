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
pred = cm.Bates_Granger_1(df_train, df_test, nu=nu)
pred = cm.Bates_Granger_2(df_train, df_test, nu=nu)
pred = cm.Bates_Granger_3(df_train, df_test, nu=nu, alpha=alpha)
pred = cm.Bates_Granger_4(df_train, df_test, W=W)
pred = cm.Bates_Granger_5(df_train, df_test, W=W)
pred = cm.Granger_Ramanathan_1(df_train, df_test)
pred = cm.Granger_Ramanathan_2(df_train, df_test)
pred = cm.Granger_Ramanathan_3(df_train, df_test)

pred1.to_latex()
type(pred1.to_latex())




# define a function for comparing forecast accuracy on a single series
def basic_forecast_accuracy_table(df, w):
    """
    The function creates latex code of the table for basic forecast comparison.

    The used measures of accuracy are Root Mean Square Error (RMSE), Mean
    Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).


    Parameters
    ----------
    df : DataFrame
        DataFrame containing the  realized values in the first column and the
        individual forecasts in the other columns
 
    w : Integer
        Integer indicating the size of the training window.
 

    Returns
    -------
    String
        String for creating the basic forecast accuracy table in latex

    """

    # number of individual forecasts and number of periods
    K = df.shape[1]-1 # -1 for realized value in the first column
    T = df.shape[0]
    
    # Accuracy measures
    measures = np.array(["RMSE", "MAE", "MASE"])
    M = measures.size
    
    # initialize forecast accuracy table
    acc_table = pd.DataFrame(data=np.full((K, M), 0),
                             columns=measures)
    
    

    return acc_table.to_latex()

df = spf_bal_RGDP_1Y

acc_table = pd.DataFrame(np.random.random((2, 2)))
acc_table.iloc[0, 0] = RMSE

w = 30
errors = np.full(w, 1)
y = np.arange(w) + 1

# END OF FILE
