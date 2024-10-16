import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf 
from matplotlib import figure
import datareader as data
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from  datetime import date, datetime

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast ')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','NFLX','TSLA')
selected_stock = st.selectbox('Select Stock for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Stock Data')
data.drop(columns=['Adj Close'], inplace=True)
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data ', xaxis_rangeslider_visible=True, title = selected_stock)
	st.plotly_chart(fig)
	
plot_raw_data()


# Function to plot data with 100-day and 200-day moving averages
def plot_data_with_moving_average():
    # Calculate moving averages
    data['100_Day_MA'] = data['Close'].rolling(window=100).mean()
    data['200_Day_MA'] = data['Close'].rolling(window=200).mean()

    fig = go.Figure()

    # Add traces for Open and Close
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))

    # Add traces for 100-day and 200-day moving averages
    fig.add_trace(go.Scatter(x=data['Date'], y=data['100_Day_MA'], name="100-Day MA", line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['200_Day_MA'], name="200-Day MA", line=dict(color='red', width=1.5)))

    fig.update_layout(title_text='Stock Data with 100-Day and 200-Day Moving Averages', xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)
plot_data_with_moving_average()


# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('PREDICTED STOCK PRICE')
st.write(forecast.tail())

st.write(f'Prediction plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
st.title("Projects")