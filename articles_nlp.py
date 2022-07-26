# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:07:57 2022

@author: aaron
"""

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from tensorflow.keras import Input, Sequential
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import os 
import re

#%% Constants

PLOT_PATH = os.path.join(os.getcwd(),'statics','model.png')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')

#%% Exploratory Data Analysis

#%% Step 1 ) Data Loading

df = pd.read_csv("https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv")

#%% Step 2) Data Inspection/Visualization

df.info()
df.describe().T
df.head()

df.duplicated().sum()
df.isna().sum()

print(df['text'][4])
print(df['text'][10])

#%% Step 3) Data Cleaning

text = df['text']
category = df['category']

for index, words in enumerate(df['text']):
    # to remove html tags
    # anything within the <> will be removed including <>
    # ? to tell re don't be greedy so it won't capture everything
    # from the first < to the last > in the document
    
    text[index] = re.sub('<.*?>','',words)   
    text[index] = re.sub('[^a-zA-Z]',' ',words).lower().split()    

text_backup = text.copy()
category_backup = category.copy()

#%% Step 4) Features Selection - text is the feature, category is the target

#%% Step 5) Data Preprocessing

vocab_size = 0
compiled = []

for i in np.arange(0,df.shape[0],1):
    compiled.extend(text[i])

vocab_size = len(set(compiled))

oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(text)

text_int = tokenizer.texts_to_sequences(text) # to convert into numbers

length_text = []

for i in range(len(text_int)):
    length_text.append(len(text_int[i]))
    # print(len(review_int[i]))

max_len = np.median([len(text_int[i]) for i in range(len(text_int))])

padded_text = pad_sequences(text_int,
                              maxlen=int(max_len),
                              padding='post',
                              truncating='post')

# y target

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

X_train, X_test, y_train, y_test = train_test_split(padded_text,category,
                                                    test_size=0.3,
                                                    random_state=123)

#%% Model Development

input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size,128)) # put embedding right after input
model.add(Bidirectional(LSTM(128,return_sequences=True))) # return sequence is true if passing to rnn,gru,ltsm
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax'))

model.summary()

plot_model(model,show_shapes=True,to_file=PLOT_PATH)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

#%% Model Training

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
early_callback = EarlyStopping(monitor='val_loss',patience=5)

hist = model.fit(X_train,y_train,
          validation_data=(X_test,y_test),
          epochs=5,
          callbacks=[tensorboard_callback])

# open anaconda prompt
# type cd "folder path to your logs file"
# type tensorboard --logdir logs

# in colab, type %load_ext tensorboard
# type %tensorboard --logdir (logs folder name that you uploaded)

#%% Model Evaluation/Analysis

keys = list(hist.history.keys())

plt.figure()
plt.plot(hist.history[keys[0]])
plt.plot(hist.history[keys[2]])
plt.xlabel('Epoch')
plt.legend(['Training '+keys[0],'Validation '+keys[0]])
plt.show()

plt.figure()
plt.plot(hist.history[keys[1]])
plt.plot(hist.history[keys[3]])
plt.xlabel('Epoch')
plt.legend(['Training '+keys[1],'Validation '+keys[1]])
plt.show()

y_pred = np.argmax(model.predict(X_test),axis=1)
y_actual = np.argmax(y_test,axis=1)

print(classification_report(y_actual, y_pred))

#%% Model Saving

# Tokenizer

token_json = tokenizer.to_json()

with open(TOKENIZER_SAVE_PATH,'w') as file:
    json.dump(token_json,file)

# OHE

with open(OHE_SAVE_PATH,'wb') as file:
    pickle.dump(ohe,file)

# MODEL

model.save(MODEL_SAVE_PATH)






















