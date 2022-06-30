import pandas as pd
import numpy as np
from matplotlib import pyplot
from dateutil.parser import parse
import datetime
import os
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
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


def get_weather_df(directory):
    # directory for sfc surface meteorogical data files

    def add_to_dict(sfc_file, weather_dict, MM, YY):
        separators = [0, 3, 8, 36, 41, 66, 72, 75, 78, 81, 87, 93, 135]
        columns = ['DD', 'HHMM', 'SKY', 'V', 'OB', 'T(C)', 'WIND DIR', 'WIND VEL', 'WIND GUST', 'PRES', 'ALT', 'REM']

        f = open(sfc_file, 'r')
        for line in f:
            if "DD HHMM SKY" not in line:
                weather_dict['MM'].append(MM)
                weather_dict['YY'].append(YY)
                for i in range(len(separators) - 1):
                    weather_dict[columns[i]].append(line[separators[i]:separators[i + 1]].strip())

    weather_dict = {
        'DD': [],
        'MM': [],
        'YY': [],
        'HHMM': [],
        'SKY': [],
        'V': [],
        'OB': [],
        'T(C)': [],
        'WIND DIR': [],
        'WIND VEL': [],
        'WIND GUST': [],
        'PRES': [],
        'ALT': [],
        'REM': [],
    }
    for file in (x for x in os.listdir(directory) if '.sfc' in x):
        add_to_dict(directory + file, weather_dict, file[:2], file[2:4])

    return pd.DataFrame(weather_dict)


def create_filtered_df(weather):
    def to_vector(deg, vel):
        try:
            deg = int(deg)
            vel = int(vel)
        except:
            return np.nan, np.nan
        else:
            rads = deg * np.pi / 180
            Wx = vel * np.cos(rads)
            Wy = vel * np.sin(rads)
            return Wx, Wy

    dic = {'Date': [], 'T(C)': [], 'Wx': [], 'Wy': [], 'Pres': [], 'Alt': []}
    for i in range(len(weather)):
        row = weather.iloc[i]

        try:
            year = int('20' + row['YY'])
            month = int(row['MM'])
            day = int(row['DD'])
            hour = int(row['HHMM'][:2])
            minute = int(row['HHMM'][2:4])
            date = datetime.datetime(year, month, day, hour, minute)
        except ValueError:
            dic['Date'].append(np.nan)
        else:
            dic['Date'].append(date)

        try:
            temp = float(row['T(C)'])
        except ValueError:
            dic['T(C)'].append(np.nan)
        else:
            dic['T(C)'].append(temp)

        try:
            pres = float(row['PRES'])
        except ValueError:
            dic['Pres'].append(np.nan)
        else:
            dic['Pres'].append(pres)

        try:
            alt = int(row['ALT'])
        except ValueError:
            dic['Alt'].append(np.nan)
        else:
            dic['Alt'].append(alt)

        try:
            Wx, Wy = to_vector(row['WIND DIR'], row['WIND VEL'])
        except (ValueError, TypeError):
            dic['Wx'].append(np.nan)
            dic['Wy'].append(np.nan)
        else:
            dic['Wx'].append(Wx)
            dic['Wy'].append(Wy)

    return pd.DataFrame(dic)


def get_snow_measurements(filename):
    snow_measurements = pd.read_csv(filename, header=1, parse_dates=True, na_values='-' )
    return snow_measurements


def just_snow_heights(snow_measurements):
    dic = {}
    dic['Date'] = snow_measurements.columns[10:]
    for i in range(len(snow_measurements)):
        dic['Station ' + snow_measurements.iloc[i][0]] = [x for x in snow_measurements.iloc[i][snow_measurements.columns[10:]]]

    return pd.DataFrame(dic)


def snow_height_deltas(snow_heights, weather):
    dic = {
        'TimeDelta': [],
        'Avg T(C)': [],
        'Avg Wx': [],
        'Avg Wy': [],
        'Avg Pres': [],
        'Avg Alt': [],
    }
    for i in range(len(snow_heights) - 1):
        sdate = parse(snow_heights['Date'][i])
        edate = parse(snow_heights['Date'][i + 1])
        date_range = weather[(weather['Date'] >= sdate) & (weather['Date'] < edate)]

        dic['TimeDelta'].append(edate - sdate)
        dic['Avg T(C)'].append(date_range['T(C)'].mean())
        dic['Avg Wx'].append(date_range['Wx'].mean())
        dic['Avg Wy'].append(date_range['Wy'].mean())
        dic['Avg Pres'].append(date_range['Pres'].mean())
        dic['Avg Alt'].append(date_range['Alt'].mean())

    for x in snow_heights.columns[1:]:
        dic[x + " Delta"] = [snow_heights.iloc[i + 1][x] - snow_heights.iloc[i][x] for i in
                             range(len(snow_heights) - 1)]

    return pd.DataFrame(dic)


def snow_height_per_year(delta):
    daily_delta = delta.rename(columns={col: col.replace('Delta', 'Height/Year') for col in delta.columns[6:]})

    for col in daily_delta.columns[6:]:
        daily_delta[col] = [daily_delta[col][i] / (daily_delta['TimeDelta'][i].days / 365.25) for i in
                            range(len(daily_delta))]

    return daily_delta.drop('TimeDelta', axis=1)


def snow_height_per_month(delta):
    monthly_delta = delta.rename(columns = {col:col.replace("Delta", "Height/Month") for col in delta.columns[6:]})
    return monthly_delta.drop('TimeDelta', axis=1)


def parse_snow_estimates(directory):
    dic = {}

    def add_to_dict(dic, filename):
        f = open(filename, 'r')
        for i, line in enumerate(f):
            if i == 0:
                dt = parse(line)
                if not 'Datetime' in dic:
                    dic['Datetime'] = []
                dic['Datetime'].append(dt)
            else:
                try:
                    station, height = line.split(" ", 1)
                    height = float(height.strip())
                except ValueError:
                    print(line)
                if not station in dic:
                    dic[station] = []
                if not height <= -99:
                    dic[station].append(height)
                else:
                    dic[station].append(np.nan)

    for year in os.listdir(directory):
        new_dir = directory + '/' + year
        for file in os.listdir(new_dir):
            add_to_dict(dic, new_dir + '/' + file)

    return pd.DataFrame(dic).rename(columns={'Datetime': 'Date'})


def snow_height_deltas2(snow_heights, weather):
    dic = {
        'TimeDelta': [],
        'Avg T(C)': [],
        'Avg Wx': [],
        'Avg Wy': [],
        'Avg Pres': [],
        'Avg Alt': [],
    }
    for i in range(len(snow_heights) - 1):
        sdate = snow_heights['Date'][i]
        edate = snow_heights['Date'][i + 1]
        date_range = weather[(weather['Date'] >= sdate) & (weather['Date'] < edate)]

        dic['TimeDelta'].append(edate - sdate)
        dic['Avg T(C)'].append(date_range['T(C)'].mean())
        dic['Avg Wx'].append(date_range['Wx'].mean())
        dic['Avg Wy'].append(date_range['Wy'].mean())
        dic['Avg Pres'].append(date_range['Pres'].mean())
        dic['Avg Alt'].append(date_range['Alt'].mean())

    for x in snow_heights.columns[1:]:
        dic[x + " Delta"] = [snow_heights.iloc[i + 1][x] - snow_heights.iloc[i][x] for i in
                             range(len(snow_heights) - 1)]
    return pd.DataFrame(dic)


def just_semi_daily_snow_heights(semi_daily_heights_raw):
    return pd.DataFrame({
        'Date':[x for x in semi_daily_heights_raw['Time (UTC)'][::-1]],
        'Heights':[float(x.split('±')[0]) if type(x) is str else np.nan for x in semi_daily_heights_raw['Snow height (cm)'][::-1] ],
    })


class Data:
    """Stores useful globals"""
    weather = get_weather_df('source_data/polar_weather/')
    filtered_weather = create_filtered_df(weather)

    snow_measurements = get_snow_measurements("source_data/SnowMeasurements.csv")
    snow_heights = just_snow_heights(snow_measurements)
    delta = snow_height_deltas(snow_heights, filtered_weather)
    delta_time = snow_height_per_year(delta)

    snow_estimates = parse_snow_estimates('source_data/monthly_snowtables_Banff_Mar2014/txt')
    delta2 = snow_height_deltas2(snow_estimates, filtered_weather)
    delta2_time = snow_height_per_month(delta2)

    models = [
        # ('Max Baseline', DummyRegressor(strategy="quantile", quantile=.99)),
        ('Mean Baseline', DummyRegressor(strategy="mean")),
        ('Linear Regression', LinearRegression()),
        ('K Neighbors', KNeighborsRegressor()),
        ('Support Vector', SVR(gamma='auto')),
        ('Random Forest', RandomForestRegressor(n_estimators=10, max_depth=10, random_state=0)),
        ('ML Perceptron', MLPRegressor(solver='lbfgs', random_state=0)),
    ]

    models_normalized = [(name, Pipeline([("scale", MinMaxScaler()), (name, model)])) for name, model in models]






