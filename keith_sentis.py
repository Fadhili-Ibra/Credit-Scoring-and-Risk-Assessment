import vaderSentiment.vaderSentiment as vader
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Paths to your text files
files = ["bbc.txt", "cnn.txt", "elpais.txt", "guardian.txt", "nbcnews.txt"]

# Initialize Sentiment Intensity Analyzer
analyzer = vader.SentimentIntensityAnalyzer()

# Store individual sentiment scores
individual_sentiments = []

# Loop through each file
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        
        # Perform sentiment analysis
        sentiment = analyzer.polarity_scores(text)
        
        # Append sentiment scores to the list
        individual_sentiments.append(sentiment)

# Aggregate sentiment scores
overall_positive = sum([s['pos'] for s in individual_sentiments]) / len(individual_sentiments)
overall_negative = sum([s['neg'] for s in individual_sentiments]) / len(individual_sentiments)
overall_neutral = sum([s['neu'] for s in individual_sentiments]) / len(individual_sentiments)

# Print overall sentiment probabilities
print(f"Overall Positive Probability: {overall_positive}")
print(f"Overall Negative Probability: {overall_negative}")
print(f"Overall Neutral Probability: {overall_neutral}")

# Plotting individual sentiment analysis
plt.figure(figsize=(12, 6))
plt.bar(files, [s['pos'] for s in individual_sentiments], color='green', label='Positive')
plt.bar(files, [s['neg'] for s in individual_sentiments], color='red', label='Negative')
plt.bar(files, [s['neu'] for s in individual_sentiments], color='blue', label='Neutral')
plt.xlabel('Files')
plt.ylabel('Probability')
plt.title('Individual Sentiment Analysis of Files')
plt.legend()
plt.show()

# Plotting overall sentiment analysis
plt.figure(figsize=(6, 4))
plt.bar(['Positive', 'Negative', 'Neutral'], [overall_positive, overall_negative, overall_neutral], color=['green', 'red', 'blue'])
plt.xlabel('Sentiment Categories')
plt.ylabel('Probability')
plt.title('Overall Sentiment Analysis of Files')
plt.xticks(rotation=45)
plt.show()