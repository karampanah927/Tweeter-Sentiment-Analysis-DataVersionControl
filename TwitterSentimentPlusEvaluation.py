import pandas as pd
from textblob import TextBlob
import re


# preprocess text data
def preprocess_text(text):
    # remove URLs
    text = re.sub(r'http\S+', '', text)
    # remove mentions
    text = re.sub(r'@\w+', '', text)
    # remove hashtags
    text = re.sub(r'#\w+', '', text)
    # remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z ]', '', text).lower()
    return text


# perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
def sentiment_result(score):
    if score>0:
        result = 'Positive'
    elif score<0:
        result = 'Negative'
    else:
        result = 'Neutral'
    return result

# read CSV file into a pandas dataframe
df = pd.read_csv('training.csv',usecols = ['rawContent'])
df_test = pd.read_csv('test.csv',usecols = ['rawContent'])
df_testGroundTruth = pd.read_csv('TweetTest.csv')

print(df.head(3))
# df = df.loc[:,'rawContent']
print(df.head(2))
df['text'] = df['rawContent'].apply(preprocess_text)
df['sentiment'] = df['text'].apply(analyze_sentiment)

df_test['text'] = df_test['rawContent'].apply(preprocess_text)
df_test['sentiment'] = df_test['text'].apply(analyze_sentiment)

# print the sentiment of each tweet
print(df['sentiment'])

# test the sentiment analysis model with a sample tweet
sample_tweet = 'I do not mind using Python for data analysis!'
sample_sentiment = analyze_sentiment(preprocess_text(sample_tweet))
print(sample_tweet)
result = sentiment_result(sample_sentiment)
print('Sentiment:', result)
#### Evaluation
T = 0
for i in range(len(df_test)):
    result = sentiment_result(df_test.iloc[i, 2])
    if result == df_testGroundTruth.iloc[i, 4]:
        T+=1
print('accuracy = ',round(T/len(df_test),2))

print("done")
