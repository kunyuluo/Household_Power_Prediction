import pickle
import pandas as pd
from Helper import split_dataset, build_model, build_model_2, plot_metrics

# Load data
# *******************************************************************************
data = pd.read_csv('household_power_consumption_day.csv', low_memory=False)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Split data into train and test set
# *******************************************************************************
train, test = split_dataset(data.values)

# Build model
# *******************************************************************************
epochs = 50
batch_size = 16
# baseline = build_model(train, test, n_input=14, n_output=7, epochs=epochs, batch_size=batch_size)
baseline = build_model_2(train, n_input=14, n_output=7, epochs=epochs, batch_size=batch_size)

model = baseline[0]
history = baseline[1]

# Save models
# *************************************************************************
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Check metrics
# *************************************************************************
# plot_metrics(history, epochs=epochs)
