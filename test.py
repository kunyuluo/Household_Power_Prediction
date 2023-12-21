import pandas as pd
from Helper import plot_variable, plot_variable_no_time
from Helper import check_linearity, check_data_distribution, check_autocorrelation
from Helper import split_dataset, to_supervised

# Load data
# *******************************************************************************
data = pd.read_csv('household_power_consumption_day.csv', low_memory=False)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# print(data.values)
# print(data.values[0])
# print(data.values.shape)
train, test = split_dataset(data.values)
# train_x, train_y = to_supervised(train, 14)
# test_x, test_y = to_supervised(test, 14)

print(train)
print(train.shape)
# print(test_x)
# print(test_x.shape)

# check_linearity(data, 'Global_active_power', 'Sub_metering_2', True)
# check_data_distribution(data, 'Sub_metering_4', False)
