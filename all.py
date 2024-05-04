import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

ticker = input("Enter the stock ticker symbol : ").upper()
import yfinance as yf

df = yf.download(tickers=ticker, start='2020-01-01', end='2024-01-01')

plt.style.use('dark_background')

plt.figure(figsize=(8, 8))
plt.title('Opening Price')
plt.plot(df['Open'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price', fontsize=18)
plt.show()
data = df.filter(['Open'])

dataset = data.values

training_data_len = math.ceil(len(dataset) * .8)

training_data_len

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(25))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=16, epochs=150)

model.save("UBER_model.h5")

test_data = scaled_data[training_data_len - 60:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

plt.style.use('dark_background')

train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('TESLA company - Model prediciton comparison')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD($)', fontsize=18)
plt.plot(train['Open'], color='red')
plt.plot(valid['Open'], color='yellow')
plt.plot(valid['predictions'], color='green')
plt.legend(['Train', 'Validation', 'predictions'], loc='lower right')
plt.show()

valid.tail(15)

stock_quote = yf.download(tickers=ticker, start='2020-01-01', end='2024-01-01')

new_df = stock_quote.filter(['Open'])
scaled_data = scaler.transform(new_df.values)

test_data = scaled_data[-60:]

predicted_prices = []

for _ in range(7):
    X_test = np.array([test_data])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)

    predicted_prices.append(pred_price[0, 0])

    test_data = np.append(test_data[1:], pred_price)

predicted_prices = scaler.inverse_transform(np.array([predicted_prices]).reshape(-1, 1))

next_7_days_dates = pd.date_range(start=stock_quote.index[-1], periods=7 + 1, freq='B')[1:]

plt.figure(figsize=(10, 6))
plt.plot(stock_quote.index, stock_quote['Open'], label='Actual Prices')
plt.plot(next_7_days_dates, predicted_prices, label='Predicted Prices', linestyle='dashed', marker='o')
plt.title(f'Predicted Stock Prices for the Next 7 Days ({ticker})')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
print(predicted_prices)