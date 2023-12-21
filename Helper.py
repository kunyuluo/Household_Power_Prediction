import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error


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


# split a univariate dataset into train/test sets
def split_dataset(data: np.ndarray):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, :]
            # x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
    return np.array(X), np.array(y)


def build_model(train, test, n_input, n_output, epochs=50, batch_size=16):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_output)
    test_x, test_y = to_supervised(test, n_input, n_output)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, return_sequences=True))
    model.add(TimeDistributed(Dense(100)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(test_x, test_y))
    return model, history


def build_model_2(train, n_input, n_output, epochs=50, batch_size=16):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_output)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history


def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def prediction(model, train, test, n_input):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = []
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
        # evaluate predictions days for each week
        # predictions = np.array(predictions)

    return np.array(predictions)


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = []
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = math.sqrt(mse)
        # store
        scores.append(rmse)

    # calculate overall RMSE
    s = 0
    score = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
            score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))

    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def plot_metrics(history, epochs: int = 25):
    acc = history.history['mape']
    val_acc = history.history['val_mape']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training MAPE')
    plt.plot(epochs_range, val_acc, label='Validation MAPE')
    plt.legend(loc='lower right')
    # plt.ylim(0, 50)
    plt.title('Training and Validation MAPE')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    # plt.ylim(0, 100)
    plt.title('Training and Validation Loss')
    plt.show()


def plot_results(test, preds, title_suffix=None, xlabel='Power Prediction'):
    """
    Plots training data in blue, actual values in red, and predictions in green, over time.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    # x = df.Close[-498:].index
    if test.shape[1] > 1:
        test = test[:, 0]

    plot_test = test[0:]
    plot_preds = preds[0:]

    # x = df[-(plot_test.shape[0] * plot_test.shape[1]):].index
    # plot_test = plot_test.reshape((plot_test.shape[0] * plot_test.shape[1], 1))
    plot_preds = plot_preds.reshape((plot_preds.shape[0] * plot_preds.shape[1], 1))

    ax.plot(plot_test, label='actual')
    ax.plot(plot_preds, label='preds')

    if title_suffix is None:
        ax.set_title('Predictions vs. Actual')
    else:
        ax.set_title(f'Predictions vs. Actual, {title_suffix}')

    ax.set_xlabel('Date')
    ax.set_ylabel(xlabel)
    ax.legend()

    plt.show()