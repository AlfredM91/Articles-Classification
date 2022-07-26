# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:07:46 2022

@author: aaron
"""

from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pickle
import json
import os

#%% Constants

BEST_MODEL_PATH = os.path.join(os.getcwd(),'saved_models','best_model.h5')
OHE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')
PLOT_PATH = os.path.join(os.getcwd(),'statics','model.png')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MODEL_PATH = os.path.join(os.getcwd(),'saved_models','model.h5')
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer.json')

#%% Classes

class ExploratoryDataAnalysis:
    
    def one_hot_encoder(self,y):
        ohe = OneHotEncoder(sparse=False)
        if y.ndim == 1:
            y = ohe.fit_transform(np.expand_dims(y,axis=-1))
        else:
            y = ohe.fit_transform(y)

        with open(OHE_PATH,'wb') as file:
            pickle.dump(OHE_PATH,file)
        return y
    
    def nlp_tokenizer_padsequences(self,df,text,padding='post',
                                   truncating='post'):
        
        compiled = []

        for i in np.arange(0,df.shape[0],1):
            compiled.extend(text[i])

        vocab_size = len(set(compiled))

        oov_token = '<OOV>'

        tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
        tokenizer.fit_on_texts(text)

        text_int = tokenizer.texts_to_sequences(text) # to convert into numbers

        max_len = np.median([len(text_int[i]) for i in range(len(text_int))])

        padded_text = pad_sequences(text_int,
                                      maxlen=int(max_len),
                                      padding=padding,
                                      truncating=truncating)
        
        token_json = tokenizer.to_json()

        with open(TOKENIZER_SAVE_PATH,'w') as file:
            json.dump(token_json,file)
        
        return padded_text
    
class ModelDevelopment:

    def dl_nlp_model(self,df,text,X_train,y_train,no_node=128,dropout_rate=0.3):
        
        input_shape = np.shape(X_train)[1:]
        
        compiled = []

        for i in np.arange(0,df.shape[0],1):
            compiled.extend(text[i])

        vocab_size = len(set(compiled))
        
        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(Embedding(vocab_size,no_node)) # put embedding right after input
        model.add(Bidirectional(LSTM(no_node,return_sequences=True))) # return sequence is true if passing to rnn,gru,ltsm
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(no_node)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(np.unique(y_train,axis=0)),activation='softmax'))

        model.summary()

        plot_model(model,show_layer_names=(True),show_shapes=True,
                   to_file=PLOT_PATH)
        
        return model
        
    def dl_model_compilation(self,model,cat_or_con):
        
        if cat_or_con=='cat':
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics='acc')
        elif cat_or_con=='con':
            model.compile(optimizer='adam',
                          loss='mse',
                          metrics='mse')
        else:
            print('Please enter either ''cat'' or ''con'' in the second argument')

    def dl_model_training(self,X_train,X_test,y_train,y_test,model,epochs=10,
                       monitor='val_loss',use_early_callback=False,
                       use_model_checkpoint=False):
        
        tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
        callbacks = [tensorboard_callback]
        
        if use_early_callback==True:
            if epochs <= 30:
                early_callback = EarlyStopping(monitor=monitor,patience=3)
                callbacks.extend([early_callback])
            else:
                early_callback = EarlyStopping(monitor=monitor,
                                               patience=np.floor(0.1*epochs))
                callbacks.extend([early_callback])
        elif use_early_callback==False:
            early_callback=None
        else:
            print('Please put only True or False for use_early_callback argument')
        
        if monitor=='val_acc':
            mode='max'
        elif monitor=='val_loss':
            mode='min'
        else:
            mode='auto'
        
        if use_model_checkpoint==True:
            model_checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor=monitor,
                                               save_best_only=(True),
                                               mode=mode,verbose=1)
            callbacks.extend([model_checkpoint])
        elif use_model_checkpoint==False:
            model_checkpoint=None
        else:
            print('Please put only True or False for use_model_checkpoint argument')
        
        hist = model.fit(X_train,y_train,epochs=epochs,verbose=1,
                         validation_data=(X_test,y_test),
                         callbacks=callbacks)
        
        model.save(MODEL_PATH)
        
        return hist
        
class ModelEvaluation:
    
    def dl_plot_hist(self,hist):
        
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
        
    def classification_report(self,X_test,y_test,best_model,ml_or_dl,
                              use_model_checkpoint=False):
        
        if use_model_checkpoint==True:
            best_model=load_model(BEST_MODEL_PATH)
        elif use_model_checkpoint==False:
            best_model=best_model
        else:
            print('Please put True or False for use_model_checkpoint argument')
            
        if ml_or_dl=='ml':
            y_pred = best_model.predict(X_test)
            y_true = y_test
        elif ml_or_dl=='dl':
            y_pred = best_model.predict(X_test)
            y_pred = np.argmax(y_pred,axis=1)
            y_true = np.argmax(y_test,axis=1)
        else:
            print('Please put either ''ml'' or ''dl'' for the ml_or_dl argument')

        cr = classification_report(y_true, y_pred)
        print(cr)
        return cr
    