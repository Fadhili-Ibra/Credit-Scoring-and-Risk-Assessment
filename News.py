import streamlit as st
import nltk
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch news table for a given ticker
def fetch_news_table(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser')
    return html.find(id='news-table')

# Function to update DataFrame based on selected ticker
def update_df(ticker):
    news_table = fetch_news_table(ticker)
    if news_table:
        parsed_data = []
        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = [elem.strip() for elem in row.td.text.split(' ') if elem.strip()]
            today_date = datetime.now().strftime('%b-%d-%y')
            if len(date_data) == 1:
                time = date_data[0]
                date = today_date
            else:
                date = date_data[0]
                time = date_data[1]
            date = today_date if date == 'Today' else date
            parsed_data.append([ticker, date, time, title])
        df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
        
        # Calculate sentiment scores
        vader = SentimentIntensityAnalyzer()
        df['Sent_Score'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
        
        return df
    else:
        st.error("Failed to fetch news table for the selected ticker.")
        return None

# Function to filter DataFrame for the past week and calculate average sentiment scores
def sentiment_analysis_weekly(df):
    df['date'] = pd.to_datetime(df['date'], format='%b-%d-%y')
    last_week = datetime.now() - timedelta(days=7)
    df_last_week = df[df['date'] >= last_week]
    daily_sentiment = df_last_week.groupby(df_last_week['date'].dt.date)['Sent_Score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Average Sentiment Score']
    return daily_sentiment

# Streamlit app
st.title("Financial News")

# Ticker selection
ticker = st.selectbox("Select Ticker", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX'])

# Update DataFrame based on selected ticker
df = update_df(ticker)
if df is not None:
    # Display DataFrame
    st.write(df)

    # Perform weekly sentiment analysis
    daily_sentiment = sentiment_analysis_weekly(df)
    
    # Display average sentiment scores for the past week as a DataFrame
    st.subheader("Average Sentiment Scores for the Past Week")
    st.write(daily_sentiment)
else:
    st.error("Failed to fetch news table for the selected ticker.")
