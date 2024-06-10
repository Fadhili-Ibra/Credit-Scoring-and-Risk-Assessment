import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.graph_objects as go
from datetime import date

# Function to display the homepage
def show_homepage():
    st.title("Stock Comparison Dashboard")
    
    st.markdown("""
    ## Welcome to the Stock Comparison Dashboard!
    
    This application allows you to:
    - Fetch and display historical stock data
    - Compare multiple stock tickers
    - Visualize stock prices over custom date ranges
    - Get financial information and news about stocks
    - Predict future stock prices using machine learning models
    
    ### Get Started
    Use the sidebar to navigate to different sections of the application or click the buttons below.
    """)
    st.image("https://example.com/stock_image.jpg", use_column_width=True)
    
    st.markdown("""
    ### Latest Updates
    - **New Feature:** Custom Date Range Selector
    - **Improved Performance:** Faster data fetching
    """)
    
    st.markdown("## Navigate to:")
    if st.button("Stock Comparison"):
        st.session_state.page = "Stock Comparison"
    if st.button("Financial Information"):
        st.session_state.page = "Financial Information"
    if st.button("Prediction"):
        st.session_state.page = "Prediction"
    if st.button("News"):
        st.session_state.page = "News"

# Function to display the stock comparison page
def show_stock_comparison():
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

# Function to display the financial information page
def show_fin_info():
    st.title('Financial Information')
    st.markdown("Here you can display detailed financial information about the selected stocks.")
    # Your financial information code here

# Function to display the prediction page
def show_prediction():
    st.title('Stock Prediction')
    st.markdown("This section will have stock price prediction features using machine learning models.")
    # Your stock prediction code here

# Function to display the news page
def show_news():
    st.title('Stock News')
    st.markdown("Get the latest news related to the selected stocks.")
    # Your stock news fetching and display code here

# Streamlit app definition
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Dashboard", "Stock Comparison", "Financial Information", "Prediction", "News"], index=["Dashboard", "Stock Comparison", "Financial Information", "Prediction", "News"].index(st.session_state.page))
    
    if selected_page == "Dashboard":
        st.session_state.page = "Dashboard"
        show_homepage()
    elif selected_page == "Stock Comparison":
        st.session_state.page = "Stock Comparison"
        show_stock_comparison()
    elif selected_page == "Financial Information":
        st.session_state.page = "Financial Information"
        show_fin_info()
    elif selected_page == "Prediction":
        st.session_state.page = "Prediction"
        show_prediction()
    elif selected_page == "News":
        st.session_state.page = "News"
        show_news()

if __name__ == "__main__":
    main()
