import streamlit as st
import nltk
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime

# Function to fetch news table for a given ticker
def fetch_news_table(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser')
    return html.find(id='news-table')

def update_df(ticker):
    news_table = fetch_news_table(ticker)
    if news_table:
        parsed_data = []
        for row in news_table.findAll('tr'):
            if row.a and row.td:
                title = row.a.text
                url = row.a.get('href')  # Correctly accessing the 'href' attribute
                date_data = [elem.strip() for elem in row.td.text.split(' ') if elem.strip()]
                today_date = datetime.now().strftime('%b-%d-%y')
                if len(date_data) == 1:
                    time = date_data[0]
                    date = today_date  # Assuming date is today if only time is given
                else:
                    date = date_data[0]
                    time = date_data[1]
                date = today_date if date.lower() == 'today' else date
                parsed_data.append([ticker, date, time, title, url])
        df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'url'])
        
        # Calculate sentiment scores
        vader = SentimentIntensityAnalyzer()
        df['Sent_Score'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
        
        return df  # Return the DataFrame for further use
    else:
        return pd.DataFrame(columns=['ticker', 'date', 'time', 'title', 'url'])  # Return an empty DataFrame if no table is found

# Streamlit app
st.title("Financial News")

# Ticker selection
ticker = st.selectbox("Select Ticker", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'NFLX'])

# Update DataFrame based on selected ticker
df = update_df(ticker)
if not df.empty:
    # Display DataFrame
    st.write(df)

    # Plot sentiment scores
    st.subheader("Sentiment Scores Distribution")
    st.bar_chart(df['Sent_Score'])
else:
    st.error("Failed to fetch news table for the selected ticker.")
