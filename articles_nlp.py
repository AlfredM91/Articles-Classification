# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:07:57 2022

@author: aaron
"""


from sklearn.model_selection import train_test_split

from articles_nlp_module import ExploratoryDataAnalysis, ModelDevelopment, ModelEvaluation
import pandas as pd
import re

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
    text[index] = re.sub('<.*?>','',words)   
    text[index] = re.sub('[^a-zA-Z]',' ',words).lower().split()    

text_backup = text.copy()
category_backup = category.copy()

#%% Step 4) Features Selection - text is the feature, category is the target

#%% Step 5) Data Preprocessing

eda = ExploratoryDataAnalysis() 

padded_text = eda.nlp_tokenizer_padsequences(df, text)

# y target

category = eda.one_hot_encoder(category)

X_train, X_test, y_train, y_test = train_test_split(padded_text,category,
                                                    test_size=0.3,
                                                    random_state=123)

#%% Model Development

md = ModelDevelopment()

model = md.dl_nlp_model(df, text, X_train, y_train)

#%% Model Compilation

md.dl_model_compilation(model, 'cat')

#%% Model Training

hist = md.dl_model_training(X_train, X_test, y_train, y_test, model, epochs=5)

# open anaconda prompt
# type cd "folder path to your logs file"
# type tensorboard --logdir logs

# in Colab, type %load_ext tensorboard
# type %tensorboard --logdir (logs folder name that you uploaded)

#%% Model Evaluation/Analysis

me = ModelEvaluation()  

# Plotting the model trained

me.dl_plot_hist(hist)

# Printing the classification report

me.classification_report(X_test, y_test, model, 'dl')


























