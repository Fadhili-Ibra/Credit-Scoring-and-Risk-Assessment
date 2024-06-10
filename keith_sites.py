import requests
from bs4 import BeautifulSoup
import io


# function to get the news content from the https://www.nbcnews.com website
def get_news_content(url, fileName):
  
    try:
        # Send a GET request to the specified URL
        response = requests.get(url)

        if response.status_code!= 200:
           print(f"Request failed with status code {response.status_code} and reason {response.reason}.")

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            
             # Extract the content within the <body> tag
            body_content = soup.body.get_text(separator='\n', strip=True)
            
            # Open (or create) the file in write mode ('w') with UTF-8 encoding
            with io.open(fileName, 'w', encoding='utf8') as file:
                # Write the content of the body tag to the file
                file.write(body_content)
            print(f"Body content saved to {fileName}")
        else:
            print("Failed to retrieve content from the URL.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to get the news content
#save_body_content_as_txt_from_url(url, filename)
get_news_content("https://www.bbc.com/news/articles/cx7768n6pnpo", "bbc.txt")

#get news from edition.cnn.com
get_news_content("https://edition.cnn.com/markets", "cnn.txt")

#get news from theguardian.com
get_news_content("https://www.theguardian.com/international", "guardian.txt")

#get news from elpais.com
get_news_content("https://elpais.com/elpais/inenglish.html", "elpais.txt")

#get news from nbcnews.com
get_news_content("https://www.nbcnews.com/business", "nbcnews.txt")

#do sentiment analysis on the content of the website
#use the following libraries
#pip install textblob
#pip install nltk
#pip install vaderSentiment
#pip install matplotlib
#pip install wordcloud



# example websites:
# https://www.theguardian.com/international
# https://edition.cnn.com/
# https://www.bbc.co.uk/news
# https://elpais.com/elpais/inenglish.html
# https://www.nbcnews.com/business


#install the required libraries
#pip install requests
#pip install beautifulsoup4




#remember the code generates a text file with the content of the website then using sentiment analysis we can determine the sentiment of the news