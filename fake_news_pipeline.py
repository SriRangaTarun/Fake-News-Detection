# Import necessary libraries

import os
import io
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from bs4.element import Comment
from bs4 import BeautifulSoup as bs

import tqdm
import zipfile
import requests
from zipfile import ZipFile
from tqdm import tqdm_notebook

import sklearn
import lightgbm as lgbm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import keras
from keras.layers import *
from keras import optimizers
from keras import activations
from keras import regularizers
from keras import initializers
from keras import constraints
from keras.models import Model
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')
path = "https://www.dropbox.com/s/2pj07qip0ei09xt/inspirit_fake_news_resources.zip?dl=1"

r = requests.get(path)
c = io.BytesIO(r.content)
z = zipfile.ZipFile(c); z.extractall(); basepath = '.'

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)
  
def prepare_data():
    train_tag_data = prepare_tag_features(train_data)
    val_tag_data = prepare_tag_features(val_data)
  
    word_index, tokenizer, train_text_data = prepare_text_data(train_data)
    _, _, val_text_data = prepare_text_data(val_data, new_tokenizer=tokenizer)
    embedding_matrix = load_glove(word_index)
  
    train_keyword_features = prepare_keyword_features(train_data)
    val_keyword_features = prepare_keyword_features(val_data)
  
    train_description_data = prepare_description_data(train_data, tokenizer=tokenizer)
    val_description_data = prepare_description_data(val_data, tokenizer=tokenizer)
  
    train_extension_data = prepare_extension_data(train_data)
    val_extension_data = prepare_extension_data(val_data)
  
    train_magic_features = prepare_magic_features(train_data)
    val_magic_features = prepare_magic_features(val_data)
  
    return tokenizer, word_index, embedding_matrix, (train_tag_data, train_text_data, train_keyword_features,
                                                     train_description_data, train_extension_data, train_magic_features),\
                                                    (val_tag_data, val_text_data, val_keyword_features,
                                                     val_description_data, val_extension_data, val_magic_features)
                                                   
    tokenizer, word_index, embedding_matrix, (train_tag_data, train_text_data, train_keyword_features,
                                              train_description_data, train_extension_data, train_magic_features),\
                                             (val_tag_data, val_text_data, val_keyword_features,
                                              val_description_data, val_extension_data, val_magic_features)\
                                             = prepare_data()
  
val_targets = np.array([data_point[2] for data_point in val_data]).reshape((len(val_data), 1))
train_targets = np.array([data_point[2] for data_point in train_data]).reshape((len(train_data), 1))
  
def neural_model():
    input_text = Input(shape=(128,), name='Text')
    input_description = Input(shape=(20,), name='Description')
    input_extension = Input(shape=(4,), name='Extension')
    input_magic = Input(shape=(5,))
    input_keywords = Input(shape=(41,), name='Keywords')
  
    embeddings_text = Embedding(len(word_index), 100, input_length=128, weights=[embedding_matrix])(input_text)
    embedding_features_text = Lambda(lambda x: K.sum(x, axis=1), name='Sum_1')(embeddings_text)
  
    embeddings_description = Embedding(len(word_index), 100, input_length=20, weights=[embedding_matrix])(input_description)
    embedding_features_description = Lambda(lambda x: K.sum(x, axis=1), name='Sum_2')(embeddings_description)
  
    features = concatenate([embedding_features_text, embedding_features_description, input_keywords, input_extension])
    features = Dropout(0.75)(features)
    features = BatchNormalization()(features)
    features = Dense(48)(features)
    features = BatchNormalization()(features)
    features = ReLU()(features)
    outputs = Dense(1, activation='sigmoid')(features)
  
    model = Model(inputs=[input_text, input_description, input_extension, input_keywords, input_magic], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model
  
model = neural_model()
model.fit(x=[train_text_data, train_description_data, train_extension_data,\
             train_keyword_features, train_magic_features], y=train_targets,\
             validation_data=([val_text_data, val_description_data,\
                               val_extension_data, val_keyword_features, val_magic_features], val_targets),
             epochs=20, batch_size=256)
            
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("model.h5")

with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
