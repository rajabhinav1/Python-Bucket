import tkinter as tk
import nltk
from nltk import word_tokenize
from tabulate import tabulate
nltk.download('twitter_samples')
# Datasets to train and test the model
from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
import pandas as pd
# Create a dataframe from positive tweets


df = pd.DataFrame(positive_tweets, columns=['Tweet'])
# Add a column to dataframe for positive sentiment value 1
df['Sentiment'] = 1
# Create a temporary dataframe for negative tweets
temp_df = pd.DataFrame(negative_tweets, columns=['Tweet'])
# Add a column to temporary dataframe for negative sentiment value 0
temp_df['Sentiment'] = 0
# Combe positive and negative tweets in one single dataframe
df = df.append(temp_df, ignore_index=True)
df = df.sample(frac = 1)
df.reset_index(drop=True, inplace=True)
# Displaying shape of dataset
print(df)
print(temp_df)
print('Dataset size:',df.shape)
print(df.groupby('Sentiment').count())

from tkinter import *

Data = (df.groupby('Sentiment').count())
Data1= (df)

root = Tk()
frame = Frame(root, width=900, height=900)
frame.pack()

lab = Label(frame,text=Data)
lab1 = Label(frame,text=Data1)
lab.pack()
lab1.pack()

root.mainloop()





