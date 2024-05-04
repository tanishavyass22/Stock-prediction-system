import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import math
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import math
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
#model for candlestick chart
def load_trained_model(symbol):
    model_path = f"C:/Users/patel/OneDrive/Documents/pythonaiml/stock1/st/model/{symbol}.h5"
    model = load_model(model_path)
    return model
def predict_next_7_day(model1, scaler, test_data):
    predicted_prices = []

    for _ in range(7):
        X_test = np.array([test_data])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        
        predicted_prices.append(pred_price[0, 0])
      
        test_data = np.append(test_data[1:], pred_price)

    predicted_prices = scaler.inverse_transform(np.array([predicted_prices]).reshape(-1, 1))
    return predicted_prices

# predict next 7 days stock prices
def predict_next_7_days(model, scaler, test_data):
    predicted_prices = []

    for _ in range(7):
        X_test = np.array([test_data])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        
        predicted_prices.append(pred_price[0, 0])
      
        test_data = np.append(test_data[1:], pred_price)

    predicted_prices = scaler.inverse_transform(np.array([predicted_prices]).reshape(-1, 1))
    return predicted_prices

#create a dataset with multiple features and outputs
def create_dataset(dataset, time_steps=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), :]
        data_x.append(a)
        data_y.append(dataset[i + time_steps, :])
    return np.array(data_x), np.array(data_y)
def load_pretrained_model(symbol):
    model_path = f"C:/Users/patel/OneDrive/Documents/pythonaiml/stock1/st/models/{symbol}_model.h5"
    model = load_model(model_path)
    return model

#train and predict stock prices
def train_and_predict_candels(ticker):
    # Download historical stock data
    df = yf.download(tickers=ticker,start='2015-01-01',end='2024-03-05')

    data = df[['Open', 'High', 'Low', 'Close']]
    dataset = data.values

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Set the training data length
    training_data_len = math.ceil(len(dataset) * 0.8)

    # Create the training data
    train_data = scaled_data[0:training_data_len, :]

    # Create the x and y training datasets
    time_steps = 10
    x_train, y_train = create_dataset(train_data, time_steps)

    # Reshape the data for LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))

    # Create the LSTM model
    model = load_trained_model(ticker)

    # Create the testing data
    test_data = scaled_data[training_data_len - time_steps:, :]

    # Create the x and y test datasets
    x_test, y_test = create_dataset(test_data, time_steps)

    # Reshape the data for LSTM model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))

    # Make predictions for the next 7 days
    predictions = []
    current_data = test_data[-time_steps:, :]
    for _ in range(7):
        pred = model.predict(np.reshape(current_data, (1, time_steps, 4)))
        predictions.append(pred[0])
        current_data = np.vstack((current_data[1:], pred[0]))

    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Display the candlestick chart using Plotly
    fig = go.Figure()

    # Actual candlesticks for the testing data
    fig.add_trace(go.Candlestick(x=df.index[training_data_len:],
                    open=df['Open'].values[training_data_len:],
                    high=df['High'].values[training_data_len:],
                    low=df['Low'].values[training_data_len:],
                    close=df['Close'].values[training_data_len:],
                    increasing_line_color='green', decreasing_line_color='red',
                    name='Actual Prices'))

    # Predicted candlesticks for the next 7 days
    fig.add_trace(go.Candlestick(x=pd.date_range(start=df.index[-1], periods=7 + 1, freq='B')[1:],
                    open=predictions[:, 0],
                    high=predictions[:, 1],
                    low=predictions[:, 2],
                    close=predictions[:, 3],
                    increasing_line_color='cyan', decreasing_line_color='gray',
                    name='Predicted Prices'))

    # Layout settings
    fig.update_layout(title=f'{ticker} - Actual vs Predicted Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Stock Price USD($)',
                      template='plotly_dark')

    # Display the plot
    st.plotly_chart(fig)

# App
def main():
    style = """
        <style>
        [data-testid="stAppViewContainer"] {
        background: url('https://wallpapercave.com/wp/wp5727299.jpg') no-repeat center center fixed;
        width: 100%;
        height: 100%;
        background-size: cover;
    }
      {
        background: url('https://img.freepik.com/free-photo/solid-gypsum-wall-textured-background_53876-101643.jpg?w=1060&t=st=1709589313~exp=1709589913~hmac=4d4f5c59faf1c52cd25e515885b694be1519b9ec19096d94b717dc566dd763b6') no-repeat center center fixed;
        width: 100%;
        height: 100%;
        background-size: cover;
    }
    .dropdown-text {
                font-size: 5em !important;
            }
           
            body {
                background: linear-gradient(to right, #f5f5f5, #d3d3d3);
                font-family: 'Times New Roman', sans-serif;
                color: #ffffff;
            }
            .title {
                color: #ffffff;
                text-align: center;
                padding: 10px;
                font-size: 5em;
                font-weight: bold;
            }
            .header {
                color:#ffffff;;
                text-align: center;
                padding: 10px;
                font-size: 1.5em;
                ;
            }
            .card-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
            }
            .card {
                background-color: #111111;
                                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 500px; /* Adjust the width as needed */
                transition: transform 0.2s;
            }

            .card:hover {
                transform: scale(1.1);
            }
            .card h4 {
                margin-bottom: 8px;
                color:#cccccc;
            }
            .card h3 {
               
                color:#cccccc;
            }
        </style>
    """
    st.markdown(style,unsafe_allow_html=True)
    st.title("Welcome To Stock Pixel!")

    # Stock symbol dropdown
    stock_symbols = [ 'AMZN','AAPL', 'MSFT', 'TSLA', 'UBER']
    selected_symbol = st.selectbox("Select Stock Symbol:", stock_symbols)

    # Predict button
    predict_button = st.button("Predict")
    model = load_pretrained_model(selected_symbol)
    scaler = MinMaxScaler(feature_range=(0, 1))
    if predict_button:
        train_and_predict_candels(selected_symbol)
        stock_data = yf.download(tickers=selected_symbol,start='2015-01-01',end='2024-03-05')

        st.markdown("<div class='header'>Historical Opening Prices:</div>", unsafe_allow_html=True)
        st.line_chart(stock_data['Open'])
        # st.markdown("<div class='header'>Model Prediction for the Next 7 Days:</div>", unsafe_allow_html=True)

        # Fit scaler with training data
        training_data = stock_data.filter(['Open']).values
        scaler.fit(training_data)

        # Prepare test data
        new_df = stock_data.filter(['Open'])
        scaled_data = scaler.transform(new_df.values)
        test_data = scaled_data[-60:]

        # Predict next 7 days
        predicted_prices = predict_next_7_days(model, scaler, test_data)

        # Display predicted prices with Streamlit charts
        next_7_days_dates = pd.date_range(start=stock_data.index[-1], periods=7 + 1, freq='B')[1:]
        predicted_chart_data = pd.DataFrame({'Date': next_7_days_dates, 'Predicted Prices': predicted_prices.flatten()})
        # Cards for predicted prices in sidebar
        st.sidebar.markdown("<div class='header'>Predicted Prices for the Next 7 Days:</div>", unsafe_allow_html=True)
        for index, row in predicted_chart_data.iterrows():
            st.sidebar.markdown(f"""
                <div class='card'>
                    <h4>{row['Date'].strftime('%Y-%m-%d')}</h4>
                    <h3>{round(row['Predicted Prices'], 2)}</h3>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
