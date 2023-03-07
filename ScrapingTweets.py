import snscrape
import emot

#libraries needed
import pandas as pd
import snscrape.modules.twitter as sntwitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

import nltk
#nltk.download('stopwords') #run once and comment it out to avoid it downloading multiple times
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
import textblob
from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS
from emot.emo_unicode import UNICODE_EMOJI

porter = PorterStemmer()

lemmatizer = WordNetLemmatizer()

from wordcloud import ImageColorGenerator
from PIL import Image

import warnings
#%matplotlib inline
import itertools

#Importing the datetime to calculate the time for scraping the 50000 tweets
from datetime import datetime



#Creating dataframe called 'data' and storing the tweets from May 1st 2021 to 30th Juy 2021 for 'Vaccine'
start_time = datetime.now()

df = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
    '"#programming"').get_items(),1000))
data = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper(
    '"#programming" since:2022-01-01 until:2023-01-20"').get_items(), 1000)) # number of tweets
end_time = datetime.now()

#Printing the time duration for scraping these tweets
print('Duration: {}'.format(end_time - start_time))

#keeping only date, id, content, user, and hashtag and stored into dataframe called 'data'

df =  data[["date","id","rawContent","renderedContent","user","hashtags","lang"]]
# training_df = data[['target', 't_id', 'created_at', 'query', 'user', 'text']]
print(df.head(5))


df.to_csv('training.csv',sep=',')