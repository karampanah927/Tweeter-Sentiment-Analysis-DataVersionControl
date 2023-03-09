import pandas as pd
from textblob import TextBlob
import re

class TwitterSentimentAnalyzer:
    # class variables and methods go here

    def __init__(self):
        self.df = pd.read_csv('training.csv', usecols=['rawContent'])
        self.df_test = pd.read_csv('test.csv', usecols=['rawContent'])
        self.df_testGroundTruth = pd.read_csv('TweetTest.csv')
        # constructor method

    # preprocess text data
    def preprocess_text(self,text):
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
    def analyze_sentiment(self,text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def train_Model(self):
        # read CSV file into a pandas dataframe


        # print(self.df.head(3))
        # # df = df.loc[:,'rawContent']
        # print(self.df.head(2))
        self.df['text'] = self.df['rawContent'].apply(self.preprocess_text)
        self.df['sentiment'] = self.df['text'].apply(self.analyze_sentiment)

        self.df_test['text'] = self.df_test['rawContent'].apply(self.preprocess_text)
        self.df_test['sentiment'] = self.df_test['text'].apply(self.analyze_sentiment)

        # print the sentiment of each tweet
        print(self.df['sentiment'])

    def eval_accuracy(self):
        #### Evaluation
        T = 0
        for i in range(len(self.df_test)):
            result = self.sentiment_result(self.df_test.iloc[i, 2])
            if result == self.df_testGroundTruth.iloc[i, 4]:
                T += 1
        print('accuracy = ', round(T / len(self.df_test), 2))

    def sentiment_result(self,score):
        if score > 0:
            result = 'Positive'
        elif score < 0:
            result = 'Negative'
        else:
            result = 'Neutral'
        return result




twitterAnalyzer = TwitterSentimentAnalyzer()


# test the sentiment analysis model with a sample tweet
sample_tweet = 'I do not mind using Python for data analysis!'
sample_sentiment = twitterAnalyzer.analyze_sentiment(twitterAnalyzer.preprocess_text(sample_tweet))
print(sample_tweet)
### our output: function: sentiment_result
result = twitterAnalyzer.sentiment_result(sample_sentiment)
print('Sentiment:', result)


print("done")
