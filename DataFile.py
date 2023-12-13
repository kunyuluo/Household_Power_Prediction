import numpy as np
import pandas as pd
import pytz
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if (values[row, col]) is None:
                values[row, col] = values[row - one_day, col]


data = pd.read_csv('household_power_consumption.txt',
                   sep=';', header=0, low_memory=False,
                   parse_dates={'datetime': [0, 1]}, index_col=['datetime'], dayfirst=True)

# mark all missing values
data.replace('?', None, inplace=True)

# make dataset numeric
data = data.astype('float32')

# fill missing
fill_missing(data.values)

# add a column for the remainder of sub metering
values = data.values
data['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])

# save updated dataset
data.to_csv('household_power_consumption.csv')

# print(data)
