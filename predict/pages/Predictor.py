import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Function to load data
start = '2010-01-01'
end = '2023-12-31'

# Ticker symbols
tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
ticker = st.sidebar.selectbox('Select Ticker', tickers, index=0)

# Preprocessing data
data = yf.download(ticker, start=start, end=end)
data2 = data.copy()
data2['% Change'] = data2['Adj Close'] / data2['Adj Close'].shift(1) - 1
data2.dropna(inplace=True)
annual_return = data2['% Change'].mean() * 252 * 100

st.header('Price Movements')
st.write(data2)
st.write('Annual Return is ', annual_return, '%')

# Function to preprocess data
def preprocess_data(data):
    data['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data.dropna(inplace=True)
    return data

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Function to build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to load stock data
def load_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Function to load news data
def load_news_data(ticker, start_date, end_date):
    news_data = pd.DataFrame({
        'Date': pd.date_range(start_date, end_date),
        'Sentiment': np.random.randn(len(pd.date_range(start_date, end_date)))
    })
    return news_data

# Plot 100-day and 200-day moving averages with buy/sell tags
st.header('Moving Averages Indicator with Buy/Sell Signals')
fig_ma, ax_ma = plt.subplots()

# Calculate moving averages
data['100 MA'] = data['Adj Close'].rolling(window=100).mean()
data['200 MA'] = data['Adj Close'].rolling(window=200).mean()

# Plot the actual price and moving averages
ax_ma.plot(data.index, data['Adj Close'], label='Actual Price')
ax_ma.plot(data.index, data['100 MA'], label='100-day MA')
ax_ma.plot(data.index, data['200 MA'], label='200-day MA')

# Find crossing points
buy_signals = []
sell_signals = []
for i in range(1, len(data)):
    if data['100 MA'].iloc[i] < data['200 MA'].iloc[i] and data['100 MA'].iloc[i-1] >= data['200 MA'].iloc[i-1]:
        sell_signals.append((data.index[i], data['Adj Close'].iloc[i]))
    elif data['100 MA'].iloc[i] > data['200 MA'].iloc[i] and data['100 MA'].iloc[i-1] <= data['200 MA'].iloc[i-1]:
        buy_signals.append((data.index[i], data['Adj Close'].iloc[i]))

# Plot buy signals
for signal in buy_signals:
    ax_ma.plot(signal[0], signal[1], 'g^', markersize=10, label='Buy Signal')

# Plot sell signals
for signal in sell_signals:
    ax_ma.plot(signal[0], signal[1], 'rv', markersize=10, label='Sell Signal')

# Ensure legend only shows one label for each signal
handles, labels = ax_ma.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_ma.legend(by_label.values(), by_label.keys())

# Set plot labels and title
ax_ma.set_title('100-day and 200-day Moving Averages with Buy/Sell Signals')
ax_ma.set_xlabel('Date')
ax_ma.set_ylabel('Price')

# Show the plot in Streamlit
st.pyplot(fig_ma)

# Main function
def main():
    st.title("Stock Price Prediction with News Sentiment")

    # User input for stock ticker and date range
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-12-31'))
    seq_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=50)

    # Load stock data
    stock_data = load_stock_data(ticker, start_date, end_date)

    # Load news data
    news_data = load_news_data(ticker, start_date, end_date)

    # Merge stock data with news data
    merged_data = pd.merge(stock_data, news_data, on='Date', how='left')

    # Preprocess data
    merged_data = preprocess_data(merged_data)

    # Create sequences for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(merged_data['Adj Close']).reshape(-1, 1))
    sequences = create_sequences(scaled_data, seq_length)

    # Split data into train and test sets
    train_size = int(len(sequences) * 0.7)
    train_data = sequences[:train_size]
    test_data = sequences[train_size:]

    # Prepare input and target variables
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Reshape input data for LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build and train LSTM model
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate model on test data
    loss = model.evaluate(X_test, y_test)
    st.write(f"Model Loss: {loss}")

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot actual vs. predicted prices
    fig, ax = plt.subplots()
    ax.plot(merged_data.index[train_size + seq_length:], merged_data['Adj Close'].values[train_size + seq_length:], label='Actual Price')
    ax.plot(merged_data.index[train_size + seq_length:], predictions, label='Predicted Price')
    ax.set_title('Actual vs. Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
