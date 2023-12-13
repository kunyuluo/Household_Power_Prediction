import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def plot_variable(df: pd.DataFrame, var_name: str = 'cp_power', is_daily: bool = True):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot()

    if is_daily:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.plot(df.index, df[var_name], color='black')
    ax.set_ylim(0, )

    plt.show()


def plot_variable_no_time(df: pd.DataFrame, var_name: str = 'cp_power'):
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot()

    x = range(len(df[var_name]))
    ax.plot(x, df[var_name], color='black')
    ax.set_ylim(0, )

    plt.show()


def check_data_distribution(df: pd.DataFrame, column_name: str = 'cp_power', log_transform: bool = False):
    """
    Check the distribution of the data.
    """
    data = df[column_name]

    if log_transform:
        data = np.log(data)

    plt.hist(data, bins=50)
    plt.title('Distribution of \'{}\''.format(column_name))
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()


def check_linearity(
        df: pd.DataFrame,
        column_name_1: str,
        column_name_2: str,
        switch_table: bool = False,
        log_transform: bool = False):
    """
    Check the linearity of the data.
    """
    data_1 = df[column_name_1]
    data_2 = df[column_name_2]

    if log_transform:
        data_1 = np.log(data_1)
        # data_2 = np.log(data_2)

    x_label = column_name_1
    y_label = column_name_2

    if switch_table:
        data_1, data_2 = data_2, data_1
        x_label, y_label = y_label, x_label

    plt.scatter(data_1, data_2, s=2)
    plt.title('Linearity between \'{}\' and \'{}\''.format(column_name_1, column_name_2))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def check_autocorrelation(df: pd.DataFrame, column_name: str = 'cp_power'):
    """
    Check the autocorrelation of the data.
    """
    data = df[column_name]

    pd.plotting.lag_plot(data)

    # plt.figure(figsize=(15, 6))
    # plt.plot(data)
    # plt.title('Autocorrelation of \'{}\''.format(column_name))
    # plt.xlabel('Time')
    # plt.ylabel(column_name)
    plt.show()