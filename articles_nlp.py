# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:07:57 2022

@author: aaron
"""

import os 
import pandas as pd

#%% Step 1 ) Data Loading

df = pd.read_csv("https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv")

#%% Step 2) Data Inspection/Visualization
df.info()
df.describe().T
df.head()

df.duplicated().sum()
df.isna().sum()

print(df['review'][4])
print(df['review'][10])

# Symbols and HTML Tags have to be removed

#%% Step 3) Data Cleaning

import re

review = df['review']
sentiment = df['sentiment']


for index, text in enumerate(df['review']):
    # to remove html tags
    # anything within the <> will be removed including <>
    # ? to tell re don't be greedy so it won't capture everything
    # from the first < to the last > in the document
    
    review[index] = re.sub('<.*?>','',text)
    review[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()    

review_backup = review.copy()
sentiment_backup = sentiment.copy()
