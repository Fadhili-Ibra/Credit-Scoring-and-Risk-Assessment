import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.graph_objects as go
from datetime import date

# Streamlit app definition
def main():
    st.title('Stock Comparison')

    # Custom Date Range Selector
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    end_date = st.date_input("End Date", value=date.today())

    # Select stock ticker(s)
    tickers = st.multiselect("Select Ticker(s)", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX'], default=['AAPL'])
    
    if not tickers:
        st.warning("Please select at least one ticker.")
        return

    # Fetch stock data
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    
    # Display stock data
    for ticker in tickers:
        st.write(f"{ticker} Stock Data")
        st.dataframe(stock_data[ticker])
    
    # Check if data is not empty
    if any(data['Close'].empty for data in stock_data.values()):
        st.error("No data available for normalization.")
        return

    # Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    closing_prices = {ticker: data['Close'].values.reshape(-1, 1) for ticker, data in stock_data.items()}
    stock_data_scaled = {ticker: scaler.fit_transform(prices) for ticker, prices in closing_prices.items()}

    # Visualization of stock data
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=stock_data[ticker].index, y=stock_data[ticker]['Close'], mode='lines', name=ticker))
    
    fig.update_layout(title="Stock Prices Over Time", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
