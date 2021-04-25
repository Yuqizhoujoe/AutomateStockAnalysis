import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, date, timedelta
from sklearn.preprocessing import RobustScaler

plt.style.use('bmh')

# Technical Analysis Library
import ta

# Neural Network Library
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

''' Loading SPY '''
stock = 'AAPL'
start_time = (datetime.today() - timedelta(days=5 * 365)).isoformat()
end_time = datetime.today().isoformat()
df = pdr.DataReader(stock, 'yahoo', start=start_time, end=end_time)
# Setting the index
df.reset_index(inplace=True)

'''Datetime conversion'''
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Dropping any NaNs
df.dropna(inplace=True)

# Converting all the column names to lowercase
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

'''Technical Indicators'''

# Adding all the indicators
df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="adj_close", volume="volume", fillna=True)

# Dropping everything else besides Adj Close and the Indicators
df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

'''Scaling'''

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()

close_scaler.fit(df[['adj_close']])

# Normalizing/Scaling the Dataframe
scaler = RobustScaler()
# remove the volatility_ui
df.drop('volatility_ui', axis=1, inplace=True)
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
print(df.index)
print(df.columns)

''' Split the Data '''

# How many periods looking back to learn
num_input_period = 90

# How many peridos to predict
num_output_period = 30

# Features
n_features = df.shape[1]

"""
Split the multivariate time sequence
"""

# Creating a list for both variables
X, y = [], []
seq = df.to_numpy()
for i in range(len(seq)):

    ## Finding the end pf the current sequence
    end = i + num_input_period
    out_end = end + num_output_period

    # Breadking out of the loop if we have exceeded that dataset's length
    if out_end > len(seq):
        break

    # Splitting the sequences into x = past prices and indicators, y = prices ahead
    seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]

    X.append(seq_x)
    y.append(seq_y)

X = np.array(X)
y = np.array(y)

# Creating the NN

# Instatiating the model
model = Sequential()

# Activation
activ = 'tanh'

# Input layer
model.add(LSTM(90,
               activation=activ,
               return_sequences=True,
               input_shape=(num_input_period, n_features)))

# Hidden layer
n_layers = 1
n_nodes = 30
drop = None
d_rate = 0.5

for x in range(1, n_layers+1):
    model.add(LSTM(n_nodes, activation=activ, return_sequences=True))

    # Addds a Dropout layer after Nth hidden alyer (the 'drop variable)
    try:
        if x % drop == 0:
            model.add(Dropout(d_rate))
    except:
        pass

# Final Hidden Layer
model.add(LSTM(60, activation=activ))

# Output layer layer
model.add(Dense(num_output_period))

# Model Summary
model.summary()

# Compiling the data with selected specification
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

## Fitting and Training
res = model.fit(X, y, epochs=50, batch_size=128, validation_split=0.1)


""" Model Validation """
# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[['adj_close']]),
                      index=df.index,
                      columns=[df.columns[0]])

# Getting a DF of the predicted values
'''
Runs a For Loop to iterate through of the DF and create predicted values for every stated interval
Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
'''
# Creating an empty DF to store the predictions
predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])
for i in range(num_input_period, len(df)-num_input_period, num_output_period):
    # Creating rolling intervals to predict off of
    x = df[-i - num_input_period:-i]

    # Predicting using rolling intervals
    yhat = model.predict(np.array(x).reshape(1, num_input_period, n_features))

    # Transforming values back to their normal prices
    yhat = close_scaler.inverse_transform(yhat)[0]

    # DF to store the values and append later, frequency uses business days
    pred_df = pd.DataFrame(yhat,
                           index=pd.date_range(start=x.index[-1], periods=len(yhat), freq="B"),
                           columns=[x.columns[0]])

    # updating the prediction DF
    predictions.update(pred_df)

# Plotting
plt.figure(figsize=(16,6))
plt.plot(predictions, label='Predicted')
plt.plot(actual, label='Actual')
plt.title(f"Predicted Vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()

# Save the Model
model.save('stock_investment/rnn_lstm_stock_price_prediction/rnn_lstm_stock_prediction')

# # Upload Model
# new_model = keras.models.load_model('stock_investment/rnn_lstm_stock_price_prediction/rnn_lstm_stock_prediction')

# Predicting off of the most recent days from the original DF
yhat = model.predict(np.array(df.tail(num_input_period)).reshape(1, num_input_period, n_features))

# Transforming the predicted values back to their original format
yhat = close_scaler.inverse_transform(yhat)[0]

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat,
                     index=pd.date_range(start=df.index[-1]+timedelta(days=1),
                                         periods=len(yhat),
                                         freq="B"),
                     columns=[df.columns[0]])

# Number of periods back to plot the actual values
pers = num_input_period

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["adj_close"]].tail(pers)),
                      index=df[['adj_close']].tail(pers).index,
                      columns=[df.columns[0]]).append(preds.head(1))

# Printing the predicted prices
print(preds)
print(actual)

# Plotting
plt.figure(figsize=(16,6))
plt.plot(actual, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("Dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()


''' Helper Functions '''
def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Split the multivariate time sequence
    """

    # Creating a list for both variables
    X, y = [], []

    for i in range(len(seq)):

        ## Finding the end pf the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out

        # Breadking out of the loop if we have exceeded that dataset's length
        if out_end > len(seq):
            break

        # Splitting the sequences into x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """

    history = results.history
    plt.figure(figsize=(16,5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()

    plt.figure(figsize=(16,5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def layer_maker(model, n_layers, n_nodes, activation, drop=None, d_rate=.5):
    """
    Creates a specified number of hidden layers for an RNN
    Optional: Adds regularization option - the dropout layer to prevent potential overfitting (if necessary)
    """

    # Creating the psrcifed number of hidden layers with the specified number of nodes
    for x in range(1, n_layers+1):
        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

        # Adds a Dropout layer after every Nth hidden layer (the drop variable)
        try:
            if x % drop == 0:
                model.add(Dropout(d_rate))
        except:
            pass
