from preprocess import *
import pandas as pd
import numpy as np
from matplotlib import pyplot
from dateutil.parser import parse
import datetime
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from preprocess import snow_height_deltas
from random import randint
import argparse


"""Snow Measurements

This script allows the user to view matplotlib graphs based on the snow height data.
Through script arguments the user can modify the data that is being graphed.

This script required pandas, sklearn, and argparse to be installed within the Python environment that it
is running on.

This script also required the "preprocess.py" module which contains the preprocessed data that this script utilizes
in its graphs.

The functions contained in this script are used for manipulating the DataFrames in the Data singleton
from preprocess.py. 

Parameters
----------
--dataset : {'yearly', 'monthly', 'daily'}, default 'yearly'
    The dataset for which the output graph represents
--model : {'all', 'rf', 'mlp', 'knn', 'lr', 'svr', 'mean'}, default All
    A string or list of strings representing which models to use in the output graphs
--compare : bool, default False
    whether or not to output a graph comparing the models used
"""


def get_X_y(df, X_cols, y_col, start_index=0):
    """Gets the X, y arrays for the machine learning models given the DataFrame and column names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to take the data,
        eg. Data.delta_time or Data.delta2_time
    X_cols: list of strings
        A list of the names of the columns that make up the X inputs.
        eg. ['Avg T(C)', 'Avg Wx', 'Avg Wy', 'Avg Pres'] or df.columns[:4]
    y_col: string
        The name of the column representing the y values.
        eg. 'Station 40B Height/Year' or df.columns[84]
    start_index: integer, default 0
        The row of the DataFrame to start extracting the data into X,y

    Returns
    -------
    (np.array, np.array)
        returns 2 numpy arrays, the first array contains arrays of integers that represent the x inputs used to predict the
        corresponding ith value of the second array. The second array contains integers representing the y values.

    """
    X = []
    y = []

    for i in range(start_index, len(df)):
        x = []
        for col in X_cols:
            x.append(df[col][i])
        X.append(x)
        y.append(df[y_col][i])

    return np.array(X), np.array(y)


def compare_models(models, X, y, title='Model Comparison', scoring='neg_mean_squared_error', useTimeSeriesSplit=False,
                   n_splits=5):
    """Outputs a graph comparing models cross validated on X, y

    Parameters
    ----------
    models : list of (string, sklearn.model)
        Takes a list of tuples containing the model name and model instance, models included are compared
        in the output graph.
        eg. [('Linear Regression', sklearn.model.LinearRegression())] or Data.models

    X: (N,N) np.array
        2 dimensional array consisting of integers
    y: (N,) np.array
        1 dimensional array consisting of integers
    title: string, default 'Model Comparison'
        title displayed above the graph
    scoring: {'r2', 'neg_mean_squared_error'}, default 'neg_mean_squared_error'
        scoring method used to compare graphs
    useTimeSeriesSplit: bool, default False
        if True conserves order of X,y when splitting for cross validation testing, use for Time Series Datasets
    n_splits: integer, default 5
        The number of splits to use for cross validation testing
    """

    names = []
    scores = []
    stdevs = []
    for name, model in models:
        if not useTimeSeriesSplit:
            dic = cross_validate(model, X, y, scoring=scoring, cv=n_splits)
        else:
            dic = cross_validate(model, X, y, scoring=scoring, cv=TimeSeriesSplit(n_splits=n_splits).split(X))

        names.append(name)
        if dic['test_score'].mean() <= 0:
            scores.append(abs(dic['test_score'].mean()))
            stdevs.append(dic['test_score'].std())

    df = pd.DataFrame({"Model": names, "Error": scores, "StDev": stdevs})
    df = df.set_index('Model')
    df.sort_values(by="Error").plot.bar(title=title)
    pyplot.show()


def plot(y, yhat, test_indices, title=None):
    predictions = [np.nan for i in y]
    for i, pred in zip(test_indices, yhat):
        predictions[i] = pred

    pd.DataFrame({
        'actual': y,
        'predictions': predictions,

    }).plot(title=title)
    pyplot.show()


def plot_integral(start_h, y, yhat, test_indices, title=None, include_diffs=False):
    predictions = [np.nan for i in range(len(y) + 1)]
    actual = [np.nan for i in range(len(y) + 1)]
    pred_diffs_only = [np.nan for i in range(len(y) + 1)]

    predictions[0] = start_h
    actual[0] = start_h
    pred_diffs_only[0] = start_h

    for i, act in zip(range(len(actual)), y):
        actual[i + 1] = actual[i] + act

    for i, pred in zip(test_indices, yhat):
        predictions[i + 1] = predictions[i] + pred

    for i, pred in zip(test_indices, yhat):
        pred_diffs_only[i + 1] = actual[i] + pred

    df = pd.DataFrame({
        # 'reference':Data.snow_estimates[Data.snow_estimates.columns[116]],  # made sure exactly the same as actual
        'actual': actual,
        'predictions': predictions,
    })
    if include_diffs:
        df['predictions_diffs_only'] = pred_diffs_only

    df.plot(title=title)
    pyplot.show()


def cross_val(model, X, y, n_splits=5, splitf=TimeSeriesSplit, name='', start_h=None, show_splits=False):
    # if integral starting is not none then it will display the snow heights rather than the change in snow heights
    # integral_starting is the starting height of the snow heights
    # if show_splits is true it shows the steps in combining all cv split results
    dic = cross_validate(model, X, y, cv=splitf(n_splits=n_splits).split(X), scoring='neg_mean_squared_error',
                         return_estimator=True)
    predictions = []
    pred_indices = []

    for i in range(len(dic['estimator'])):
        indices = [x for x in splitf(n_splits=n_splits).split(X)]
        train_indices = indices[i][0]
        test_indices = indices[i][1]
        yhat = dic['estimator'][i].predict(X[test_indices])

        predictions = predictions + list(yhat)
        pred_indices = pred_indices + list(test_indices)

        if show_splits:
            plot(y, yhat, test_indices, name)
            print(mean_squared_error(y[test_indices], yhat))

    if show_splits:
        print(dic['test_score'])

    print(f"MSE: {abs(round(sum(dic['test_score']) / len(dic['test_score']), 5))}")

    if start_h is not None:
        plot_integral(start_h, y, predictions, pred_indices, name)
    else:
        plot(y, predictions, pred_indices, name)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="choose between {'yearly', 'monthly', 'daily'}, default is yearly")
    args = parser.parse_args()

    X, y = get_X_y(Data.delta_time, Data.delta_time.columns[:4], Data.delta_time.columns[84])
    compare_models(Data.models, X, y, 'Yearly Snow Heights')
    # X, y = get_X_y(Data.delta2_time, Data.delta2_time.columns[:4], Data.delta2_time.columns[120])
    # compare_models(Data.models, X, y, 'Monthly Snow Heights')


if __name__ == '__main__':
    main()
