import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf 
import plotly.express as px 


default_ticker = "TSLA"
default_start_date = pd.to_datetime("2020-01-01")
default_end_date = pd.Timestamp.now().date()

st.title('Stock Dashboard')

#ticker = st.sidebar.text_input('Ticker', default_ticker)
tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'BTC-USD']
ticker = st.sidebar.selectbox('Select Ticker', tickers, index=0)  

start_date = st.sidebar.date_input('Start Date', default_start_date)
end_date =  st.sidebar.date_input('End Date', default_end_date)

data = yf.download(ticker, start=start_date, end=end_date)
fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)
st.plotly_chart(fig)

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    data2 = data 
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) -1 
    data2.dropna(inplace = True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is ', annual_return,'%')
    
from alpha_vantage.fundamentaldata import FundamentalData 
with fundamental_data:
    key = '9C2OWRXPCKQHEBVS'
    fd = FundamentalData(key, output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd. get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    #st.write('Fundamental')
    
    st.subheader('Income Statements')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns= list(income_statement.T.iloc[0])
    st.write(is1)
    
    st.subheader('Cash Flow Statements')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)
    
    from stocknews import StockNews
with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news = False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
    st.write(f'news Sentiment {news_sentiment}')
