# -*- coding: utf-8 -*-
"""
COMBINATION METHODS module

This module contains definitions of forecast combinations methods, which are
used to produce the combine forecasts.

The first input should be a DataFrame with the training data of the following
form: the first column contains the realized values, the other columns contain
individual forecasts, the row index corresponds to time axis. The second input
should be a DataFrame with the test data (i.e. all columns contain individual
forecasts). The remaining input should be parameters relevant for each
respective method.

The output is a DataFrame with one column containing combined forecasts
(predictions) of length corresponding to the length of the supplied test
DataFrame.

"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA

"""
SIMPLE METHODS

"""


def Equal_Weights(df_test):
    """
    For this combining method, no training set is actually necessary. The
    predictions can be generated by averaging the individual forecasts supplied
    for testing.

    """

    df_pred = pd.DataFrame({"Equal Weights": np.mean(df_test, axis=1)})

    return df_pred


def Bates_Granger_1(df_train, df_test, nu):
    """
    This method combines the individual forecasts linearly, using the length
    of the training window nu.

    Firstly, the vector of squared forecast errors is calculated for each
    of the individual forecasts. Secondly, the vector of weights for prediction
    is calculated based on the error vectors. Lastly, the weights are used to
    combine forecasts and produce prediction for testing dataset.

    """

    # number of periods
    T = df_train.shape[0]

    if nu > T:
        raise ValueError('Parameter nu must be <= length of training sample')

    # forecast errors
    errors = df_train.iloc[:, 1:].subtract(df_train.iloc[:, 0], axis=0)
    sq_errors = errors**2

    # combining weights
    nominator = 1 / sq_errors.iloc[sq_errors.shape[0]-nu:, :].sum(axis=0)
    denominator = nominator.sum()
    comb_w = nominator / denominator

    # predictions
    df_pred = pd.DataFrame({"Bates-Granger (1)": df_test.dot(comb_w)})

    return df_pred


def Bates_Granger_2(df_train, df_test, nu):
    """
    This method combines the individual forecasts linearly, using the length
    of the training window nu.

    Firstly, the vector of forecast errors is calculated for each
    of the individual forecasts. Secondly, the estimate of the covariance
    matrix sigma is calculated. Lastly, the weights are used to combine
    forecasts and produce prediction for testing dataset. The weights lying
    out of the [0,1] interval are replaced by the appropriate end points.

    In order for covariance matrix sigma to be invertible, it is necesarry,
    that the parameter nu is greater or equal to the number of forecast to
    be combined.

    """
    # number of individual forecasts and number of periods
    K = df_test.shape[1]
    T = df_train.shape[0]

    if nu > T:
        raise ValueError('Parameter nu must be <= length of training sample')

    # check whether there is enough observations, so sigma is invertible
    if nu < K:
        raise ValueError('Parameter nu must be >= no. of individual forecasts')

    # forecast errors
    errors = df_train.iloc[:, 1:].subtract(df_train.iloc[:, 0], axis=0)

    # initialize the covariance matrix sigma
    sigma = np.full((K, K), fill_value=0, dtype=float)

    # fill the covariance matrix sigma
    for i in range(K):

        for j in range(K):

            sigma[i, j] = np.dot(errors.iloc[errors.shape[0]-nu:, i],
                                 errors.iloc[errors.shape[0]-nu:, j]) / nu

    # combining weights
    nominator = np.linalg.solve(sigma, np.full(K, fill_value=1))
    denominator = np.dot(np.full(K, fill_value=1), nominator)
    comb_w = nominator / denominator

    # censoring the combining weights
    for i in range(K):
        if comb_w[i] < 0:
            comb_w[i] = 0
        if comb_w[i] > 1:
            comb_w[i] = 1

    # rescale the weights so that their sum equals 1
    comb_w = comb_w/comb_w.sum()

    # predictions
    df_pred = pd.DataFrame({"Bates-Granger (2)": df_test.dot(comb_w)})

    return df_pred


def Bates_Granger_3(df_train, df_test, nu, alpha):
    """

    This method convexely combines the weights obtained from Bates-Granger
    method (1) and the preceeding weights, assuming that omega_{i,1} = 1 / K
    for all i. The combinations is determined by the parameter alpha.

    """

    # number of individual forecasts and number of periods
    K = df_test.shape[1]
    T = df_train.shape[0]

    if nu > T:
        raise ValueError('Parameter nu must be <= length of training sample')

    if nu < K:
        raise ValueError('Parameter nu must be >= no. of individual forecasts')

    # matrix of combination weights (t = 1,...,T, T+1), T+1 is for the final c.
    mat_comb_w = np.full((T+1, K), fill_value=0, dtype=float)

    # initialize with equal weights
    mat_comb_w[:nu, :] = np.full(K, fill_value=1 / K, dtype=float)

    # roll over the training period and calculate the combining weights
    for i in range(nu, T+1):

        # compute the weights using Bates-Granger method 1
        # forecast errors
        errors = df_train.iloc[:i, 1:].subtract(df_train.iloc[:i, 0], axis=0)
        sq_errors = errors**2

        # combining weights
        nominator = 1 / sq_errors.iloc[sq_errors.shape[0]-nu:, :].sum(axis=0)
        denominator = nominator.sum()
        method_1_comb_w = nominator / denominator

        # calculate and store the combined combining weights
        mat_comb_w[i, :] = alpha*mat_comb_w[i-1, :] + (1-alpha)*method_1_comb_w

    # final combining weights (weights for period T+1 = index T)
    comb_w = mat_comb_w[T, :]

    # predictions
    df_pred = pd.DataFrame({"Bates-Granger (3)": df_test.dot(comb_w)})

    return df_pred


def Bates_Granger_4(df_train, df_test, W):
    """
    This method combines the individual forecasts linearly, and uses the weight
    parameter W to assign more weight to recent errors.

    Firstly, the vector of squared forecast errors is calculated for each
    of the individual forecasts. Secondly, the vector of weights for prediction
    is calculated based on the error vectors. Lastly, the weights are used to
    combine forecasts and produce prediction for testing dataset.

    """

    # number of individual forecasts
    T = df_train.shape[0]

    # forecast errors
    errors = df_train.iloc[:, 1:].subtract(df_train.iloc[:, 0], axis=0)
    sq_errors = errors**2

    # exponential error weights
    error_w = np.full(T, fill_value=W, dtype=float)**(np.arange(T)+1)

    # combining weights
    nominator = 1 / np.dot(error_w, sq_errors)
    denominator = nominator.sum()
    comb_w = nominator / denominator

    # predictions
    df_pred = pd.DataFrame({"Bates-Granger (4)": df_test.dot(comb_w)})

    return df_pred


def Bates_Granger_5(df_train, df_test, W):
    """
    This method resembles the second method, except that it uses the weight
    parameter W to assign more weight to recent errors.

    Firstly, the vector of forecast errors is calculated for each
    of the individual forecasts. Secondly, the estimate of the covariance
    matrix sigma is calculated. Lastly, the weights are used to combine
    forecasts and produce prediction for testing dataset. The weights lying
    out of the [0,1] interval are replaced by the appropriate end points.

    In order for covariance matrix sigma to be invertible, it is necesarry,
    that the length of the training sample is greater or equal to the number
    of forecast to be combined.

    """
    # number of individual forecasts and number of periods
    K = df_test.shape[1]
    T = df_train.shape[0]

    # check whether there is enough observations, so sigma is invertible
    if K > T:
        raise ValueError('No. forecasts must be <= length of training sample')

    # forecast errors
    errors = df_train.iloc[:, 1:].subtract(df_train.iloc[:, 0], axis=0)

    # initialize the covariance matrix sigma
    sigma = np.full((K, K), fill_value=0, dtype=float)

    # exponential error weights
    error_w = np.full(T, fill_value=W, dtype=float)**(np.arange(T)+1)
    error_w_sum = error_w.sum()

    # fill the covariance matrix sigma
    for i in range(K):

        for j in range(K):

            # elements in sigma matrix are weighted with W
            sigma[i, j] = np.dot(error_w*errors.iloc[:, i],
                                 errors.iloc[:, j]) / error_w_sum

    # combining weights
    nominator = np.linalg.solve(sigma, np.full(K, fill_value=1))
    denominator = np.dot(np.full(K, fill_value=1), nominator)
    comb_w = nominator / denominator

    # censoring the combining weights
    for i in range(K):
        if comb_w[i] < 0:
            comb_w[i] = 0
        if comb_w[i] > 1:
            comb_w[i] = 1

    # rescale the weights so that their sum equals 1
    comb_w = comb_w/comb_w.sum()

    # predictions
    df_pred = pd.DataFrame({"Bates-Granger (5)": df_test.dot(comb_w)})

    return df_pred


def Granger_Ramanathan_1(df_train, df_test):
    """
    This method combines the individual forecasts linearly, and uses the weight
    parameters estimated using OLS.

    In the first method, the weights are estimated using unconstrained
    regression without an intercept. The weights are then used to combine
    forecasts and produce predictions for testing dataset.

    """
    # define y, F
    y = df_train.iloc[:, 0]
    F = df_train.iloc[:, 1:]

    # create linear regression object
    lin_reg = linear_model.LinearRegression(fit_intercept=False)

    # fit the model
    lin_reg.fit(F, y)

    # compute the combining weights
    beta_hat = lin_reg.coef_

    # predictions
    df_pred = pd.DataFrame({"Granger-Ramanathan (1)": df_test.dot(beta_hat)})

    return df_pred


def Granger_Ramanathan_2(df_train, df_test):
    """
    This method combines the individual forecasts linearly, and uses the weight
    parameters estimated using OLS.

    In the second method, the weights are estimated using contrained regression
    (sum of coefficients equals one) without an intercept. As described in the
    thesis I use the other, computationally equivalent way, of estimating the
    weights. Firstly, I compute the y_star and F_star by subtracting the
    last individual forecast. Secondly I estimate the beta_star and then I
    compute the last weight as 1-beta_star. The weights are then used to
    combine forecasts and produce predictions for testing dataset.

    """

    # number of individual forecasts
    K = df_test.shape[1]

    # define y_star, F_star
    y_star = df_train.iloc[:, 0] - df_train.iloc[:, K]
    F_star = df_train.iloc[:, 1:K].subtract(df_train.iloc[:, K], axis=0)

    # create linear regression object
    lin_reg = linear_model.LinearRegression(fit_intercept=False)

    # fit the model
    lin_reg.fit(F_star, y_star)

    # compute the combining weights
    beta_star_hat = lin_reg.coef_
    beta_K = 1 - beta_star_hat.sum()
    beta_hat = np.append(beta_star_hat, beta_K)

    # predictions
    df_pred = pd.DataFrame({"Granger-Ramanathan (2)": df_test.dot(beta_hat)})

    return df_pred


def Granger_Ramanathan_3(df_train, df_test):
    """
    This method combines the individual forecasts linearly, and uses the weight
    parameters estimated using OLS.

    In the third method, the weights are estimated using unconstrained
    regression with an intercept. The weights and intercept are then used to
    combine forecasts and produce predictions for testing dataset.

    """

    # define y, F
    y = df_train.iloc[:, 0]
    F = df_train.iloc[:, 1:]

    # create linear regression object
    lin_reg = linear_model.LinearRegression(fit_intercept=True)

    # fit the model
    lin_reg.fit(F, y)

    # store the estimated beta and alpha
    alpha_hat = lin_reg.intercept_
    beta_hat = lin_reg.coef_

    # predictions
    df_pred = pd.DataFrame({"Granger-Ramanathan (3)":
                            alpha_hat + df_test.dot(beta_hat)})

    return df_pred


def AFTER(df_train, df_test, lambd):
    """
    This method combines the individual forecasts linearly, using the tuning
    parameter alpha.

    Firstly, the vector of squared forecast errors is calculated for each
    of the individual forecasts. Secondly, the vector of weights for prediction
    is calculated based on the error vectors. Lastly, the weights are used to
    combine forecasts and produce prediction for testing dataset.

    """

    # forecast errors
    errors = df_train.iloc[:, 1:].subtract(df_train.iloc[:, 0], axis=0)
    sq_errors = errors**2

    # combining weights
    nominator = np.exp((-lambd) * sq_errors.sum(axis=0))
    denominator = nominator.sum()
    comb_w = nominator / denominator

    # predictions
    df_pred = pd.DataFrame({"AFTER": df_test.dot(comb_w)})

    return df_pred


def Median_Forecast(df_test):
    """
    For this combining method, no training set is actually necessary. The
    predictions can obtained by as a median from the forecasts supplied
    for testing.

    """

    df_pred = pd.DataFrame(
                {"Median Forecast": np.median(df_test, axis=1)},
                index=df_test.index
                )

    return df_pred


def Trimmed_Mean_Forecast(df_test, alpha):
    """
    For this combining method, no training set is actually necessary. The
    predictions can obtained by as an alpha-trimmed mean from the forecasts
    supplied for testing.

    """

    # number of individual forecasts
    K = df_test.shape[1]

    # number values to be removed
    r = np.floor(alpha*K).astype(int)

    # trimmed testing set
    df_test_trim = np.sort(df_test)[:, r:(K-r)]

    # predictions
    df_pred = pd.DataFrame(
                {"Trimmed Mean Forecast": np.mean(df_test_trim, axis=1)},
                index=df_test.index
                )

    return df_pred


def PEW(df_train, df_test):
    """
    The projection on the equall weights: regress the variable to be forecast
    on the average of individual forecasts (with intercept).

    """

    # define y, f_bar
    y = df_train.iloc[:, 0]
    f_bar = np.mean(df_train.iloc[:, 1:], axis=1).values.reshape(-1, 1)

    # create linear regression object
    lin_reg = linear_model.LinearRegression(fit_intercept=True)

    # fit the model
    lin_reg.fit(f_bar, y)

    # store the estimated beta and alpha
    alpha_hat = lin_reg.intercept_
    beta_hat = lin_reg.coef_

    # predictions
    df_pred = pd.DataFrame({"PEW":
                            alpha_hat + beta_hat*np.mean(df_test, axis=1)})

    return df_pred


"""
FACTOR ANALYTIC METHODS

"""


def Principal_Component_Forecast(df_train, df_test, prcomp):
    """
    The principal component (factor analytic) combination method. The number
    of used components depends on the parameter "prcomp" = {single, AIC, BIC}

    """

    # number of periods
    T = df_train.shape[0]

    # compute second moment matrix of forecasts
    sec_mom_mat = np.dot(np.transpose(df_train.iloc[:, 1:]),
                         df_train.iloc[:, 1:])

    if (prcomp == "single"):

        # estimate the common factor mu using principal components
        pca = PCA(n_components=1).fit(sec_mom_mat)
        lambda_hat = np.transpose(pca.components_)
        mu_hat = np.dot(df_train.iloc[:, 1:], lambda_hat)

        # define y
        y = df_train.iloc[:, 0]

        # create linear regression object
        lin_reg = linear_model.LinearRegression(fit_intercept=False)

        # fit the model
        lin_reg.fit(mu_hat, y)

        # store the estimated beta
        beta_hat = lin_reg.coef_

        # predictions
        df_pred = pd.DataFrame(
                {"Principal Component Forecast":
                    np.dot(np.dot(df_test, lambda_hat), beta_hat)},
                index=df_test.index)

    if (prcomp == "AIC"):

        # initialize final beta coefficients and AIC
        final_beta_hat = np.nan
        final_lambda_hat = np.nan
        final_AIC = np.inf

        for i in range(4):
            # number of principal components, +1 for python
            prcomp_no = i + 1

            # estimate the common factor mu using principal components
            pca = PCA(n_components=prcomp_no).fit(sec_mom_mat)
            lambda_hat = np.transpose(pca.components_)
            mu_hat = np.dot(df_train.iloc[:, 1:], lambda_hat)

            # define y
            y = df_train.iloc[:, 0]

            # create linear regression object
            lin_reg = linear_model.LinearRegression(fit_intercept=False)

            # fit the model
            lin_reg.fit(mu_hat, y)

            # store the estimated beta
            beta_hat = lin_reg.coef_

            # AIC calculation
            res = y - np.dot(mu_hat, beta_hat)
            res_sq = res**2
            sigma_sq = sum(res_sq)/T
            P = prcomp_no + 1  # total number of parameters, including sigma
            AIC = T*np.log(sigma_sq) + 2*P

            # change final beta and AIC, if there is improvement in AIC
            if AIC < final_AIC:
                final_AIC = AIC
                final_beta_hat = beta_hat
                final_lambda_hat = lambda_hat

        # predictions
        df_pred = pd.DataFrame(
                {"Principal Component Forecast (AIC)":
                    np.dot(np.dot(df_test, final_lambda_hat), final_beta_hat)},
                index=df_test.index)

    if (prcomp == "BIC"):

        # initialize final beta coefficients, eigenvectors and BIC
        final_beta_hat = np.nan
        final_lambda_hat = np.nan
        final_BIC = np.inf

        for i in range(4):
            # number of principal components, +1 for python
            prcomp_no = i + 1

            # estimate the common factor mu using principal components
            pca = PCA(n_components=prcomp_no).fit(sec_mom_mat)
            lambda_hat = np.transpose(pca.components_)
            mu_hat = np.dot(df_train.iloc[:, 1:], lambda_hat)

            # define y
            y = df_train.iloc[:, 0]

            # create linear regression object
            lin_reg = linear_model.LinearRegression(fit_intercept=False)

            # fit the model
            lin_reg.fit(mu_hat, y)

            # store the estimated beta
            beta_hat = lin_reg.coef_

            # BIC calculation
            res = y - np.dot(mu_hat, beta_hat)
            res_sq = res**2
            sigma_sq = sum(res_sq)/T
            P = prcomp_no + 1  # total number of parameters, including sigma
            BIC = T*np.log(sigma_sq) + P*np.log(T)

            # change the final coefficients, if there is improvement in BIC
            if BIC < final_BIC:
                final_BIC = BIC
                final_beta_hat = beta_hat
                final_lambda_hat = lambda_hat

        # predictions
        df_pred = pd.DataFrame(
                {"Principal Component Forecast (BIC)":
                    np.dot(np.dot(df_test, final_lambda_hat), final_beta_hat)},
                index=df_test.index)

    return df_pred


"""
SHRINKING METHODS

"""


def Empirical_Bayes_Estimator(df_train, df_test):
    """
    The empirical bayes estimator of combining weights. Bayesian linear
    regression model with the prior specified in Diebold & Pauli (1990).

    """

    # number of individual forecasts and number of periods
    K = df_test.shape[1]
    T = df_train.shape[0]

    # define the prior weights (simple average, with intercept equal zero)
    beta_0 = np.append(np.array(0), np.full(K, fill_value=1/K, dtype=float))

    # design matrix (intercept + individual forecasts)
    F = df_train.iloc[:, 1:].values
    F = np.insert(F, 0, 1, axis=1)

    # define y (observed values)
    y = df_train.iloc[:, 0].values

    # OLS weights
    beta_hat = np.dot(np.linalg.inv(np.dot(np.transpose(F), F)),
                      np.dot(np.transpose(F), y))

    # sigma
    sigma_hat_sq = np.dot(
            np.transpose(y - np.dot(F, beta_hat)),
            y - np.dot(F, beta_hat)
            ) / T

    # tau
    num = np.dot(np.transpose(beta_hat - beta_0), beta_hat - beta_0)
    denum = np.trace(np.linalg.inv(np.dot(np.transpose(F), F)))
    tau_hat_sq = num / denum - sigma_hat_sq

    # combining weights
    shrinkage = 1 - sigma_hat_sq / (sigma_hat_sq + tau_hat_sq)
    beta_1_hat = beta_0 + shrinkage*(beta_hat - beta_0)

    # predictions
    df_pred = pd.DataFrame(
            {"Empirical Bayes Estimator":
                beta_1_hat[0] + np.dot(df_test, beta_1_hat[1:])},
            index=df_test.index)

    return df_pred


def Kappa_Shrinkage(df_train, df_test, kappa):
    """
    Shrinkage of the combining weights from OLS without intercept towards
    the equal weights. The amount of shrinkage is driven by parameter kappa.

    """

    # number of individual forecasts and number of periods
    K = df_test.shape[1]
    T = df_train.shape[0]

    # define the prior weights (simple average, with intercept equal zero)
    beta_0 = np.full(K, fill_value=1/K, dtype=float)

    # design matrix (intercept + individual forecasts)
    F = df_train.iloc[:, 1:].values

    # define y (observed values)
    y = df_train.iloc[:, 0].values

    # OLS weights (without an intercept)
    beta_hat = np.dot(np.linalg.inv(np.dot(np.transpose(F), F)),
                      np.dot(np.transpose(F), y))

    # shrinkage weight
    lambd = max([0, 1 - kappa * (K / (T - 1 - K))])

    # combining weights
    comb_w = lambd*beta_hat + (1-lambd)*beta_0

    # predictions
    df_pred = pd.DataFrame({"Kappa-Shrinkage": df_test.dot(comb_w)})

    return df_pred


# THE END OF MODULE
