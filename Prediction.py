import pickle
import pandas as pd
from matplotlib import pyplot
from Helper import prediction, split_dataset, evaluate_forecasts, summarize_scores, plot_results

# Load the model
# *************************************************************************
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
# *******************************************************************************
data = pd.read_csv('household_power_consumption_day.csv', low_memory=False)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Split data into train and test set
# *******************************************************************************
train, test = split_dataset(data.values)
predictions = prediction(model, train, test, 14)
predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])

test = test[:, :, 0]
# test = test.reshape(test.shape[0] * test.shape[1], 1)

# print(test[:, 0])
print(test.shape)
# print(predictions)
print(predictions.shape)

score, scores = evaluate_forecasts(test, predictions)
summarize_scores('lstm', score, scores)
# plot_results(test, predictions)

# plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
# pyplot.plot(days, scores, marker='o', label='lstm')
# pyplot.show()
