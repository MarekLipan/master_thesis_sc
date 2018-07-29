"""
HYPOTHESES TESTING

This script defines the Diebold-Mariano test and uses it to test the hypotheses
described in the thesis.

"""

import pandas as pd
import numpy as np
import accuracy_measures as am
import forecast_tables as ft
import statsmodels.formula.api as smf
from pylatex import Table, Tabular, MultiColumn, MultiRow
from pylatex.utils import NoEscape

# data path
data_path = "C:/Users/Marek/Dropbox/Master_Thesis/Data/"


def DM_test(errors_1, errors_2):
    """
    The Diebold-Mariano for a given pair of forecast errors. It uses
    HAC (4 lag) and a quadratic loss. It outputs the p-value from test
    with appropriate symbol: asterisk if forecast_2 is significantly better,
    dagger if forecast 1 is significantly better

    Parameters
    ----------
    errors_1 : NumpyArray
        Vector of forecast errors.

    errors_1 : NumpyArray
        Vector of forecast errors.

    Returns
    -------
    p_val : Str
        P-value from the DM test
    """

    loss_diff = pd.DataFrame({"d": errors_1**2 - errors_2**2})

    if (loss_diff.values == 0).all():
        p_val = "X"
    else:
        results = smf.ols('d ~ 1', data=loss_diff).fit(
                cov_type='HAC', cov_kwds={'maxlags': 4})

        t_val = results.tvalues[0]
        p_val_num = results.pvalues[0]

        if p_val_num <= 0.01:
            sig = 3
        elif p_val_num <= 0.05:
            sig = 2
        elif p_val_num <= 0.1:
            sig = 1
        else:
            sig = 0

        if sig == 0:
            p_val = NoEscape("$" + "{:.2f}".format(p_val_num) + "$")
        elif t_val > 0:
            p_val = NoEscape("$" + "{:.2f}".format(p_val_num) + "^{"+sig*"*"+"}$")
        else:
            p_val = NoEscape("$" + "{:.2f}".format(p_val_num) + "^{"+sig*"\dagger"+"}$")

    return p_val


comb_methods = ['Equal Weights', 'Bates-Granger (1)', 'Bates-Granger (2)',
       'Bates-Granger (3)', 'Bates-Granger (4)', 'Bates-Granger (5)',
       'Granger-Ramanathan (1)', 'Granger-Ramanathan (2)',
       'Granger-Ramanathan (3)', 'AFTER', 'Median Forecast',
       'Trimmed Mean Forecast', 'PEW', 'Principal Component Forecast',
       'Principal Component Forecast (AIC)',
       'Principal Component Forecast (BIC)', 'Empirical Bayes Estimator',
       'Kappa-Shrinkage', 'Two-Step Egalitarian LASSO',
       'BMA (Marginal Likelihood)', 'BMA (Predictive Likelihood)', 'ANN',
       'EP-NN', 'Bagging', 'Componentwise Boosting', 'AdaBoost',
       'c-APM (Constant)', 'c-APM (Q-learning)', 'Market for Kernels']


################################################
# HYPOTHESIS 1) equal weights vs. all - ECB SPF#
################################################

hyp_1_spf = pd.DataFrame(data=np.full((len(comb_methods), 18), "", dtype=str),
                         index=comb_methods)

# w = 25
j=0
df_name = "spf_bal_RGDP_1Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=1
df_name = "spf_bal_RGDP_2Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=2
df_name = "spf_bal_HICP_1Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=3
df_name = "spf_bal_HICP_2Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=4
df_name = "spf_bal_UNEM_1Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=5
df_name = "spf_bal_UNEM_2Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# w = 35
j=6
df_name = "spf_bal_RGDP_1Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=7
df_name = "spf_bal_RGDP_2Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=8
df_name = "spf_bal_HICP_1Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=9
df_name = "spf_bal_HICP_2Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=10
df_name = "spf_bal_UNEM_1Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=11
df_name = "spf_bal_UNEM_2Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# w = 45
j=12
df_name = "spf_bal_RGDP_1Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=13
df_name = "spf_bal_RGDP_2Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=14
df_name = "spf_bal_HICP_1Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=15
df_name = "spf_bal_HICP_2Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=16
df_name = "spf_bal_UNEM_1Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=17
df_name = "spf_bal_UNEM_2Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_1_spf.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: equal weights against the remaining combinations of forecasts from the ECB SPF")
tabl.append(NoEscape('\label{tab: DM_hyp_1}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 9*"|cc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiColumn(6, align='c', data="w = 25"),
              MultiColumn(6, align='|c', data="w = 35"),
              MultiColumn(6, align='|c', data="w = 45")))

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + [
        MultiColumn(2, align='c', data="RGDP"),
        MultiColumn(2, align='|c', data="HICP"),
        MultiColumn(2, align='|c', data="UNEM")]+2*[
        MultiColumn(2, align='|c', data="RGDP"),
        MultiColumn(2, align='|c', data="HICP"),
        MultiColumn(2, align='|c', data="UNEM")])

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + 9*["1Y", "2Y"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [hyp_1_spf.index[0]] + list(hyp_1_spf.iloc[0, :]))
for i in range(1, 13):
    tabr.add_row([""] + [hyp_1_spf.index[i]] + list(hyp_1_spf.iloc[i, :]))

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [hyp_1_spf.index[13]] + list(hyp_1_spf.iloc[13, :]))
for i in range(14, 16):
    tabr.add_row([""] + [hyp_1_spf.index[i]] + list(hyp_1_spf.iloc[i, :]))

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [hyp_1_spf.index[16]] + list(hyp_1_spf.iloc[16, :]))
for i in range(17, 19):
    tabr.add_row([""] + [hyp_1_spf.index[i]] + list(hyp_1_spf.iloc[i, :]))

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [hyp_1_spf.index[19]] + list(hyp_1_spf.iloc[19, :]))
for i in range(20, 21):
    tabr.add_row([""] + [hyp_1_spf.index[i]] + list(hyp_1_spf.iloc[i, :]))

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [hyp_1_spf.index[21]] + list(hyp_1_spf.iloc[21, :]))
for i in range(22, 26):
        tabr.add_row([""] + [hyp_1_spf.index[i]] + list(hyp_1_spf.iloc[i, :]))

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [hyp_1_spf.index[26]] + list(hyp_1_spf.iloc[26, :]))
for i in range(27, 29):
    tabr.add_row([""] + [hyp_1_spf.index[i]] + list(hyp_1_spf.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_1")


#####################################################
# HYPOTHESIS 2) all vs. Market for Kernels - ECB SPF#
#####################################################

hyp_2_spf = pd.DataFrame(data=np.full((len(comb_methods), 18), "", dtype=str),
                         index=comb_methods)

# w = 25
j=0
df_name = "spf_bal_RGDP_1Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=1
df_name = "spf_bal_RGDP_2Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=2
df_name = "spf_bal_HICP_1Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=3
df_name = "spf_bal_HICP_2Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=4
df_name = "spf_bal_UNEM_1Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=5
df_name = "spf_bal_UNEM_2Y_25"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# w = 35
j=6
df_name = "spf_bal_RGDP_1Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=7
df_name = "spf_bal_RGDP_2Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=8
df_name = "spf_bal_HICP_1Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=9
df_name = "spf_bal_HICP_2Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=10
df_name = "spf_bal_UNEM_1Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=11
df_name = "spf_bal_UNEM_2Y_35"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# w = 45
j=12
df_name = "spf_bal_RGDP_1Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_RGDP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=13
df_name = "spf_bal_RGDP_2Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_RGDP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=14
df_name = "spf_bal_HICP_1Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_HICP_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=15
df_name = "spf_bal_HICP_2Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_HICP_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=16
df_name = "spf_bal_UNEM_1Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_UNEM_1Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=17
df_name = "spf_bal_UNEM_2Y_45"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = spf_bal_UNEM_2Y.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_2_spf.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: Market for Kernels against the remaining combinations of forecasts from the ECB SPF")
tabl.append(NoEscape('\label{tab: DM_hyp_2}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 9*"|cc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiColumn(6, align='c', data="w = 25"),
              MultiColumn(6, align='|c', data="w = 35"),
              MultiColumn(6, align='|c', data="w = 45")))

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + [
        MultiColumn(2, align='c', data="RGDP"),
        MultiColumn(2, align='|c', data="HICP"),
        MultiColumn(2, align='|c', data="UNEM")]+2*[
        MultiColumn(2, align='|c', data="RGDP"),
        MultiColumn(2, align='|c', data="HICP"),
        MultiColumn(2, align='|c', data="UNEM")])

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + 9*["1Y", "2Y"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [hyp_2_spf.index[0]] + list(hyp_2_spf.iloc[0, :]))
for i in range(1, 13):
    tabr.add_row([""] + [hyp_2_spf.index[i]] + list(hyp_2_spf.iloc[i, :]))

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [hyp_2_spf.index[13]] + list(hyp_2_spf.iloc[13, :]))
for i in range(14, 16):
    tabr.add_row([""] + [hyp_2_spf.index[i]] + list(hyp_2_spf.iloc[i, :]))

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [hyp_2_spf.index[16]] + list(hyp_2_spf.iloc[16, :]))
for i in range(17, 19):
    tabr.add_row([""] + [hyp_2_spf.index[i]] + list(hyp_2_spf.iloc[i, :]))

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [hyp_2_spf.index[19]] + list(hyp_2_spf.iloc[19, :]))
for i in range(20, 21):
    tabr.add_row([""] + [hyp_2_spf.index[i]] + list(hyp_2_spf.iloc[i, :]))

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [hyp_2_spf.index[21]] + list(hyp_2_spf.iloc[21, :]))
for i in range(22, 26):
        tabr.add_row([""] + [hyp_2_spf.index[i]] + list(hyp_2_spf.iloc[i, :]))

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [hyp_2_spf.index[26]] + list(hyp_2_spf.iloc[26, :]))
for i in range(27, 29):
    tabr.add_row([""] + [hyp_2_spf.index[i]] + list(hyp_2_spf.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_2")




#############################################
# HYPOTHESIS 3) equal weights vs. all - RVOL#
#############################################

# TU + FV
hyp_3_TU_FV = pd.DataFrame(data=np.full((len(comb_methods), 18), "", dtype=str),
                           index=comb_methods)

# TU
# h = 1
j=0
df_name = "ind_fcts_1_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=1
df_name = "ind_fcts_1_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=2
df_name = "ind_fcts_1_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 5
j=3
df_name = "ind_fcts_5_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=4
df_name = "ind_fcts_5_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=5
df_name = "ind_fcts_5_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 22
j=6
df_name = "ind_fcts_22_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=7
df_name = "ind_fcts_22_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=8
df_name = "ind_fcts_22_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# FV
# h = 1
j=9
df_name = "ind_fcts_1_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=10
df_name = "ind_fcts_1_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=11
df_name = "ind_fcts_1_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 5
j=12
df_name = "ind_fcts_5_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=13
df_name = "ind_fcts_5_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=14
df_name = "ind_fcts_5_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 22
j=15
df_name = "ind_fcts_22_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=16
df_name = "ind_fcts_22_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=17
df_name = "ind_fcts_22_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TU_FV.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: equal weights against the remaining combinations of forecasts of U.S. Treasury futures RVOL")
tabl.append(NoEscape('\label{tab: DM_hyp_3_TU_FV}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 6*"|ccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiColumn(9, align='c', data="TU (2 Year)"),
              MultiColumn(9, align='|c', data="FV (5 Year)")))

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + [
        MultiColumn(3, align='c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")]+[
        MultiColumn(3, align='|c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")])

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + 6*["w=100", "w=200", "w=500"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [hyp_3_TU_FV.index[0]] + list(hyp_3_TU_FV.iloc[0, :]))
for i in range(1, 13):
    tabr.add_row([""] + [hyp_3_TU_FV.index[i]] + list(hyp_3_TU_FV.iloc[i, :]))

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [hyp_3_TU_FV.index[13]] + list(hyp_3_TU_FV.iloc[13, :]))
for i in range(14, 16):
    tabr.add_row([""] + [hyp_3_TU_FV.index[i]] + list(hyp_3_TU_FV.iloc[i, :]))

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [hyp_3_TU_FV.index[16]] + list(hyp_3_TU_FV.iloc[16, :]))
for i in range(17, 19):
    tabr.add_row([""] + [hyp_3_TU_FV.index[i]] + list(hyp_3_TU_FV.iloc[i, :]))

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [hyp_3_TU_FV.index[19]] + list(hyp_3_TU_FV.iloc[19, :]))
for i in range(20, 21):
    tabr.add_row([""] + [hyp_3_TU_FV.index[i]] + list(hyp_3_TU_FV.iloc[i, :]))

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [hyp_3_TU_FV.index[21]] + list(hyp_3_TU_FV.iloc[21, :]))
for i in range(22, 26):
        tabr.add_row([""] + [hyp_3_TU_FV.index[i]] + list(hyp_3_TU_FV.iloc[i, :]))

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [hyp_3_TU_FV.index[26]] + list(hyp_3_TU_FV.iloc[26, :]))
for i in range(27, 29):
    tabr.add_row([""] + [hyp_3_TU_FV.index[i]] + list(hyp_3_TU_FV.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_3_TU_FV")


# TY + US
hyp_3_TY_US = pd.DataFrame(data=np.full((len(comb_methods), 18), "", dtype=str),
                           index=comb_methods)

# TY
# h = 1
j=0
df_name = "ind_fcts_1_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=1
df_name = "ind_fcts_1_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=2
df_name = "ind_fcts_1_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 5
j=3
df_name = "ind_fcts_5_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=4
df_name = "ind_fcts_5_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=5
df_name = "ind_fcts_5_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 22
j=6
df_name = "ind_fcts_22_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=7
df_name = "ind_fcts_22_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=8
df_name = "ind_fcts_22_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# US
# h = 1
j=9
df_name = "ind_fcts_1_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=10
df_name = "ind_fcts_1_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=11
df_name = "ind_fcts_1_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 5
j=12
df_name = "ind_fcts_5_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=13
df_name = "ind_fcts_5_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=14
df_name = "ind_fcts_5_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

# h = 22
j=15
df_name = "ind_fcts_22_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=16
df_name = "ind_fcts_22_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])

j=17
df_name = "ind_fcts_22_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 0]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_3_TY_US.iloc[i, j] = DM_test(benchmark_errors, error_mat[:, i])


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: equal weights against the remaining combinations of forecasts of U.S. Treasury futures RVOL")
tabl.append(NoEscape('\label{tab: DM_hyp_3_TY_US}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 6*"|ccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiColumn(9, align='c', data="TY (10 Year)"),
              MultiColumn(9, align='|c', data="US (30 Year)")))

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + [
        MultiColumn(3, align='c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")]+[
        MultiColumn(3, align='|c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")])

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + 6*["w=100", "w=200", "w=500"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [hyp_3_TY_US.index[0]] + list(hyp_3_TY_US.iloc[0, :]))
for i in range(1, 13):
    tabr.add_row([""] + [hyp_3_TY_US.index[i]] + list(hyp_3_TY_US.iloc[i, :]))

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [hyp_3_TY_US.index[13]] + list(hyp_3_TY_US.iloc[13, :]))
for i in range(14, 16):
    tabr.add_row([""] + [hyp_3_TY_US.index[i]] + list(hyp_3_TY_US.iloc[i, :]))

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [hyp_3_TY_US.index[16]] + list(hyp_3_TY_US.iloc[16, :]))
for i in range(17, 19):
    tabr.add_row([""] + [hyp_3_TY_US.index[i]] + list(hyp_3_TY_US.iloc[i, :]))

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [hyp_3_TY_US.index[19]] + list(hyp_3_TY_US.iloc[19, :]))
for i in range(20, 21):
    tabr.add_row([""] + [hyp_3_TY_US.index[i]] + list(hyp_3_TY_US.iloc[i, :]))

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [hyp_3_TY_US.index[21]] + list(hyp_3_TY_US.iloc[21, :]))
for i in range(22, 26):
        tabr.add_row([""] + [hyp_3_TY_US.index[i]] + list(hyp_3_TY_US.iloc[i, :]))

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [hyp_3_TY_US.index[26]] + list(hyp_3_TY_US.iloc[26, :]))
for i in range(27, 29):
    tabr.add_row([""] + [hyp_3_TY_US.index[i]] + list(hyp_3_TY_US.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_3_TY_US")

##################################################
# HYPOTHESIS 4) all vs. Market for Kernels - RVOL#
##################################################

# TU + FV
hyp_4_TU_FV = pd.DataFrame(data=np.full((len(comb_methods), 18), "", dtype=str),
                           index=comb_methods)

# TU
# h = 1
j=0
df_name = "ind_fcts_1_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=1
df_name = "ind_fcts_1_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=2
df_name = "ind_fcts_1_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 5
j=3
df_name = "ind_fcts_5_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=4
df_name = "ind_fcts_5_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=5
df_name = "ind_fcts_5_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 22
j=6
df_name = "ind_fcts_22_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=7
df_name = "ind_fcts_22_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=8
df_name = "ind_fcts_22_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# FV
# h = 1
j=9
df_name = "ind_fcts_1_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=10
df_name = "ind_fcts_1_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=11
df_name = "ind_fcts_1_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 5
j=12
df_name = "ind_fcts_5_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=13
df_name = "ind_fcts_5_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=14
df_name = "ind_fcts_5_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 22
j=15
df_name = "ind_fcts_22_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=16
df_name = "ind_fcts_22_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=17
df_name = "ind_fcts_22_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TU_FV.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: Market for Kernels against the remaining combinations of forecasts of U.S. Treasury futures RVOL")
tabl.append(NoEscape('\label{tab: DM_hyp_4_TU_FV}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 6*"|ccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiColumn(9, align='c', data="TU (2 Year)"),
              MultiColumn(9, align='|c', data="FV (5 Year)")))

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + [
        MultiColumn(3, align='c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")]+[
        MultiColumn(3, align='|c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")])

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + 6*["w=100", "w=200", "w=500"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [hyp_4_TU_FV.index[0]] + list(hyp_4_TU_FV.iloc[0, :]))
for i in range(1, 13):
    tabr.add_row([""] + [hyp_4_TU_FV.index[i]] + list(hyp_4_TU_FV.iloc[i, :]))

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [hyp_4_TU_FV.index[13]] + list(hyp_4_TU_FV.iloc[13, :]))
for i in range(14, 16):
    tabr.add_row([""] + [hyp_4_TU_FV.index[i]] + list(hyp_4_TU_FV.iloc[i, :]))

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [hyp_4_TU_FV.index[16]] + list(hyp_4_TU_FV.iloc[16, :]))
for i in range(17, 19):
    tabr.add_row([""] + [hyp_4_TU_FV.index[i]] + list(hyp_4_TU_FV.iloc[i, :]))

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [hyp_4_TU_FV.index[19]] + list(hyp_4_TU_FV.iloc[19, :]))
for i in range(20, 21):
    tabr.add_row([""] + [hyp_4_TU_FV.index[i]] + list(hyp_4_TU_FV.iloc[i, :]))

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [hyp_4_TU_FV.index[21]] + list(hyp_4_TU_FV.iloc[21, :]))
for i in range(22, 26):
        tabr.add_row([""] + [hyp_4_TU_FV.index[i]] + list(hyp_4_TU_FV.iloc[i, :]))

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [hyp_4_TU_FV.index[26]] + list(hyp_4_TU_FV.iloc[26, :]))
for i in range(27, 29):
    tabr.add_row([""] + [hyp_4_TU_FV.index[i]] + list(hyp_4_TU_FV.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_4_TU_FV")


# TY + US
hyp_4_TY_US = pd.DataFrame(data=np.full((len(comb_methods), 18), "", dtype=str),
                           index=comb_methods)

# TY
# h = 1
j=0
df_name = "ind_fcts_1_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=1
df_name = "ind_fcts_1_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=2
df_name = "ind_fcts_1_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 5
j=3
df_name = "ind_fcts_5_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=4
df_name = "ind_fcts_5_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=5
df_name = "ind_fcts_5_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 22
j=6
df_name = "ind_fcts_22_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=7
df_name = "ind_fcts_22_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=8
df_name = "ind_fcts_22_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# US
# h = 1
j=9
df_name = "ind_fcts_1_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=10
df_name = "ind_fcts_1_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=11
df_name = "ind_fcts_1_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 5
j=12
df_name = "ind_fcts_5_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=13
df_name = "ind_fcts_5_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=14
df_name = "ind_fcts_5_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 22
j=15
df_name = "ind_fcts_22_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=16
df_name = "ind_fcts_22_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=17
df_name = "ind_fcts_22_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - fcts_table.values[:, :]
for i in range(len(comb_methods)):
    hyp_4_TY_US.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: Market for Kernels against the remaining combinations of forecasts of U.S. Treasury futures RVOL")
tabl.append(NoEscape('\label{tab: DM_hyp_4_TY_US}'))
# create tabular object
tabr = Tabular(table_spec="c|l" + 6*"|ccc")
tabr.add_hline()
tabr.add_hline()
# header row
tabr.add_row((MultiRow(3, data="Class"),
              MultiRow(3, data="Forecast Combination Method"),
              MultiColumn(9, align='c', data="TY (10 Year)"),
              MultiColumn(9, align='|c', data="US (30 Year)")))

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + [
        MultiColumn(3, align='c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")]+[
        MultiColumn(3, align='|c', data="h = 1"),
        MultiColumn(3, align='|c', data="h = 5"),
        MultiColumn(3, align='|c', data="h = 22")])

tabr.add_hline(start=3, end=20, cmidruleoption="lr")

tabr.add_row(2*[""] + 6*["w=100", "w=200", "w=500"])

tabr.add_hline()
# fill in the rows of tabular
# Simple
tabr.add_row([MultiRow(13, data="Simple")] + [hyp_4_TY_US.index[0]] + list(hyp_4_TY_US.iloc[0, :]))
for i in range(1, 13):
    tabr.add_row([""] + [hyp_4_TY_US.index[i]] + list(hyp_4_TY_US.iloc[i, :]))

tabr.add_hline()
# Factor Analytic
tabr.add_row([MultiRow(3, data="Factor An.")] + [hyp_4_TY_US.index[13]] + list(hyp_4_TY_US.iloc[13, :]))
for i in range(14, 16):
    tabr.add_row([""] + [hyp_4_TY_US.index[i]] + list(hyp_4_TY_US.iloc[i, :]))

tabr.add_hline()
# Shrinkage
tabr.add_row([MultiRow(3, data="Shrinkage")] + [hyp_4_TY_US.index[16]] + list(hyp_4_TY_US.iloc[16, :]))
for i in range(17, 19):
    tabr.add_row([""] + [hyp_4_TY_US.index[i]] + list(hyp_4_TY_US.iloc[i, :]))

tabr.add_hline()
# BMA
tabr.add_row([MultiRow(2, data="BMA")] + [hyp_4_TY_US.index[19]] + list(hyp_4_TY_US.iloc[19, :]))
for i in range(20, 21):
    tabr.add_row([""] + [hyp_4_TY_US.index[i]] + list(hyp_4_TY_US.iloc[i, :]))

tabr.add_hline()
# Alternative
tabr.add_row([MultiRow(5, data="Alternative")] + [hyp_4_TY_US.index[21]] + list(hyp_4_TY_US.iloc[21, :]))
for i in range(22, 26):
        tabr.add_row([""] + [hyp_4_TY_US.index[i]] + list(hyp_4_TY_US.iloc[i, :]))

tabr.add_hline()
# APM
tabr.add_row([MultiRow(3, data="APM")] + [hyp_4_TY_US.index[26]] + list(hyp_4_TY_US.iloc[26, :]))
for i in range(27, 29):
    tabr.add_row([""] + [hyp_4_TY_US.index[i]] + list(hyp_4_TY_US.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_4_TY_US")



###################################################################
# HYPOTHESIS 5) individual forecasts vs. Market for Kernels - RVOL#
###################################################################

ind_fcts = list(ind_fcts_1_TU)[1:]

hyp_5_100 = pd.DataFrame(data=np.full((len(ind_fcts), 12), "", dtype=str),
                        index=ind_fcts)
hyp_5_200 = pd.DataFrame(data=np.full((len(ind_fcts), 12), "", dtype=str),
                        index=ind_fcts)
hyp_5_500 = pd.DataFrame(data=np.full((len(ind_fcts), 12), "", dtype=str),
                        index=ind_fcts)

# h = 1
j=0
df_name = "ind_fcts_1_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=1
df_name = "ind_fcts_1_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=2
df_name = "ind_fcts_1_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=3
df_name = "ind_fcts_1_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_1_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_1_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_1_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 5
j=4
df_name = "ind_fcts_5_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=5
df_name = "ind_fcts_5_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=6
df_name = "ind_fcts_5_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=7
df_name = "ind_fcts_5_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_5_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_5_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_5_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

# h = 22
j=8
df_name = "ind_fcts_22_TU_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_TU_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_TU_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_TU.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=9
df_name = "ind_fcts_22_FV_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_FV_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_FV_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_FV.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=10
df_name = "ind_fcts_22_TY_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_TY_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_TY_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_TY.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

j=11
df_name = "ind_fcts_22_US_100"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_100.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_US_200"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_200.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)

df_name = "ind_fcts_22_US_500"
fcts_table = pd.read_pickle(data_path + "Multiproc/MP_" + df_name + ".pkl")
benchmark_errors = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0] - fcts_table.values[:, 28]
error_mat = ind_fcts_22_US.values[(-fcts_table.shape[0]):, 0][:, np.newaxis] - ind_fcts_22_US.values[(-fcts_table.shape[0]):, 1:]
for i in range(len(ind_fcts)):
    hyp_5_500.iloc[i, j] = DM_test(error_mat[:, i], benchmark_errors)


# create table object
tabl = Table()
tabl.add_caption("P-values from the DM test of equal forecast accuracy: Market for Kernels against the individual forecasts of U.S. Treasury futures RVOL")
tabl.append(NoEscape('\label{tab: DM_hyp_5}'))
# create tabular object
tabr = Tabular(table_spec="c|l|cccc|cccc|cccc")
tabr.add_hline()
tabr.add_hline()
# header row

tabr.add_row((MultiRow(2, data="Future"), MultiRow(2, data="Volatility Model"),
              MultiColumn(4, align='c|', data="h = 1"),
              MultiColumn(4, align='c|', data="h = 5"),
              MultiColumn(4, align='c', data="h = 22")))
tabr.add_hline(start=3, end=14, cmidruleoption="lr")
tabr.add_row([""]*2 + ["TU", "FV", "TY", "US"]*3)
tabr.add_hline()
# fill in the rows of tabular
tabr.add_row([MultiRow(12, data="w = 100")] + [hyp_5_100.index[0]] + list(hyp_5_100.iloc[0, :]))
for i in range(1, 12):
    tabr.add_row([""] + [hyp_5_100.index[i]] + list(hyp_5_500.iloc[i, :]))

tabr.add_hline()
tabr.add_row([MultiRow(12, data="w = 200")] + [hyp_5_200.index[0]] + list(hyp_5_200.iloc[0, :]))
for i in range(1, 12):
    tabr.add_row([""] + [hyp_5_200.index[i]] + list(hyp_5_500.iloc[i, :]))

tabr.add_hline()
tabr.add_row([MultiRow(12, data="w = 500")] + [hyp_5_500.index[0]] + list(hyp_5_500.iloc[0, :]))
for i in range(1, 12):
    tabr.add_row([""] + [hyp_5_500.index[i]] + list(hyp_5_500.iloc[i, :]))

# end of table
tabr.add_hline()
# add tabular to table
tabl.append(tabr)
# export the table
tab_path = "C:/Users/Marek/Dropbox/Master_Thesis/Latex/Tables/"
tabl.generate_tex(tab_path + "DM_hyp_5")


###############
# END OF FILE #
###############
