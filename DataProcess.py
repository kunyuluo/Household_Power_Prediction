import pandas as pd
from Helper import plot_variable, plot_variable_no_time
from Helper import check_linearity, check_data_distribution, check_autocorrelation

# Load data
# *******************************************************************************
data = pd.read_csv('household_power_consumption.csv', low_memory=False)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data.resample('15min').sum()
# data = data.groupby(pd.Grouper(freq='B')).sum()
print(data)
# check_linearity(data, 'Global_active_power', 'Sub_metering_2', True)
# check_data_distribution(data, 'Sub_metering_4', False)
