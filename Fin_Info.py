import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    negative_returns = excess_returns[excess_returns < 0]
    return excess_returns.mean() / negative_returns.std() * np.sqrt(252)

# Define the Streamlit app
def main():
    st.title('Stock Financial Information')

    # Dropdown menu for selecting data type (financial statements or time series data)
    selected_data_type = st.selectbox('Select Data Type', ['Financial Statements', 'Time Series Data'])

    if selected_data_type == 'Financial Statements':
        # Dropdown menu for selecting financial data function
        selected_function = st.selectbox('Select Financial Data Function', ['get_income_stmt', 'get_balance_sheet', 'get_cashflow'])

        # Dropdown menu for selecting stock symbol(s)
        selected_stocks = st.multiselect('Select Stocks', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX'])
    
        if not selected_stocks:
            st.warning('Please select at least one stock.')
        else:
            for stock_symbol in selected_stocks:
                st.write(f"Financial Data for {stock_symbol}")
                stock = yf.Ticker(stock_symbol)
                
                # Fetch the requested financial data function
                financial_data_function = getattr(stock, selected_function)
                
                # Get the financial data
                financial_data = financial_data_function()
                
                if financial_data is not None:
                    # Display the financial data as a table
                    st.table(financial_data)
                else:
                    st.warning(f"No {selected_function} data available for {stock_symbol}")

    elif selected_data_type == 'Time Series Data':
        # Dropdown menu for selecting time series data type
        selected_time_series_type = st.selectbox('Select Time Series Data Type', ['History', 'Dividends', 'Splits', 'Returns'])

        # Dropdown menu for selecting stock symbol
        selected_stock = st.selectbox('Select Stock', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX'])

        st.write(f"Time Series Data for {selected_stock}")
        stock = yf.Ticker(selected_stock)
        
        if selected_time_series_type == 'History':
            # Get historical market data
            hist = stock.history(period="1mo")
            st.table(hist)
        
        elif selected_time_series_type == 'Dividends':
            # Get dividends data
            dividends = stock.dividends
            st.table(dividends)
        
        elif selected_time_series_type == 'Splits':
            # Get splits data
            splits = stock.splits
            st.table(splits)
        
        elif selected_time_series_type == 'Returns':
            # Get historical market data
            hist = stock.history(period="1y")  # Get 1 year of data for return calculations
            
            if hist is not None:
                # Calculate Simple Returns
                hist['Simple Return'] = hist['Close'].pct_change()

                # Calculate Log Returns
                hist['Log Return'] = np.log(hist['Close'] / hist['Close'].shift(1))

                # Calculate average return and volatility
                average_return = hist['Simple Return'].mean() * 252
                volatility = hist['Simple Return'].std() * np.sqrt(252)

                st.write("Stock Returns")
                st.table(hist[['Close', 'Simple Return', 'Log Return']].dropna())

                # Calculate Sharpe and Sortino Ratios
                risk_free_rate = 0  # Assume risk-free rate is 0 for simplicity, can be changed
                sharpe_ratio = calculate_sharpe_ratio(hist['Simple Return'].dropna(), risk_free_rate)
                sortino_ratio = calculate_sortino_ratio(hist['Simple Return'].dropna(), risk_free_rate)

                st.write(f"Average Annual Return: {average_return:.2%}")
                st.write(f"Annual Volatility: {volatility:.2%}")
                st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                st.write(f"Sortino Ratio: {sortino_ratio:.2f}")

                # Plotting the Simple Returns
                st.write("Simple Returns Over Time")
                plt.figure(figsize=(10, 6))
                plt.plot(hist.index, hist['Simple Return'], label='Simple Return')
                plt.title(f'{selected_stock} Simple Returns')
                plt.xlabel('Date')
                plt.ylabel('Simple Return')
                plt.legend()
                st.pyplot(plt)

                # Plotting the Log Returns
                st.write("Log Returns Over Time")
                plt.figure(figsize=(10, 6))
                plt.plot(hist.index, hist['Log Return'], label='Log Return')
                plt.title(f'{selected_stock} Log Returns')
                plt.xlabel('Date')
                plt.ylabel('Log Return')
                plt.legend()
                st.pyplot(plt)
            else:
                st.warning(f"No historical data available for {selected_stock}")

if __name__ == "__main__":
    main()
