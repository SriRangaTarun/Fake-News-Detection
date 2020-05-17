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

# Define path for loading data and load train/val datasets

parser = argparse.ArgumentParser()
parser.add_argument('glove_path')
args = parser.parse_args(); glove_path = args.glove_path
path = "https://www.dropbox.com/s/2pj07qip0ei09xt/inspirit_fake_news_resources.zip?dl=1"

r = requests.get(path)
c = io.BytesIO(r.content)
z = zipfile.ZipFile(c); z.extractall(); basepath = '.'

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
    train_data, val_data = pickle.load(f)
    
# Define helper functions to calculate tag, extension, headline, and magic features
  
# Get tag features
def prepare_tag_features(data):
    vectorizer = CountVectorizer()
  
    raw_tag_data = []
    for data_point in tqdm_notebook(data):
      raw_tag_data.append(' '.join([tag.name for tag in bs(data_point[1]).find_all()]))
    
    tag_counts = vectorizer.fit_transform(raw_tag_data)
    tag_counts = tag_counts.toarray()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(tag_counts)
    tag_counts = scaler.transform(tag_counts)
    return tag_counts

# Get visible tags on page
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

# Get visible text from HTML
def text_from_html(body):
    soup = bs(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)
    
# Get binarykeyword features
def prepare_keyword_features(data):
    keywords = ['support', 'day','because','in',\
                'but','will','work','jew','proven','house','people','state','percent',\
                'sanders','Nazi','share','said', 'Obama', 'Trump', 'h1', 'h2', 'title','h3',
                'kill','die','offcial','election', 'elections', 'government','!','/', 'secret',\
                'controversial','confidential', 'censor','censored','posted','html','<','claims','>']
  
    keyword_features = []
  
    for data_point in tqdm_notebook(data):
      keyword_feature = []
      html = data_point[1]
      for keyword in keywords:
        if keyword in html:
          keyword_feature.append(1)
        else:
          keyword_feature.append(0)
      keyword_features.append(keyword_feature)
    
    keyword_features = np.array(keyword_features)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(keyword_features)
    keyword_features = scaler.transform(keyword_features)
    return keyword_features

# Load GloVe embeddings
def load_glove(word_index):
    EMBEDDING_FILE = glove_path
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:100]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(200000, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in tqdm_notebook(word_index.items()):
        if i >= 200000: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector
            
    return embedding_matrix

# Prepare text data with tokenizer
def prepare_text_data(data, new_tokenizer=None):
    text_data = []
    for data_point in tqdm_notebook(data):
      text_data.append(text_from_html(data_point[1]).strip().replace(u'\xa0', u' '))
  
    if new_tokenizer is None:
      tokenizer = Tokenizer(num_words=200000)
      is_train = True
    else:
      tokenizer = new_tokenizer
      is_train=False

    if is_train: tokenizer.fit_on_texts(text_data)
    text_data = tokenizer.texts_to_sequences(text_data)
    text_data = pad_sequences(text_data, maxlen=128)
    return tokenizer.word_index, tokenizer, text_data

# Get extension features
def prepare_extension_data(data):
    extension_data = []
  
    for datapoint in tqdm_notebook(data):
      extension_list = []
      url = datapoint[0]
    
      def add_extension(extension):
          extension_list.append(int(url.endswith(extension)))
      
      extensions = ['.com', '.org', '.net', '.gov']
    
      for extension in extensions: add_extension(extension)
      extension_data.append(extension_list)
    
    return np.array(extension_data)

# Get magic keyword features
def prepare_magic_features(data):
    magic_features = []
    shady_chars = ['$', '!', ',', '%', '~', '*', '&', '^', '#', ')', '(', '@']
    fake_chars = ['obama', 'trump', 'Obama', 'Trump', 'h1', 'h2', 'title']
    ad_chars = ['ad', 'Ad', 'pop', 'Pop', 'bygoogle', 'Bygoogle', 'byGoogle', 'ByGoogle']
    bg_chars = ['bgcolor="#']
  
    for data_point in tqdm_notebook(data):
        magic_feature = []
        url = data_point[0]
        html = data_point[1]
    
        magic_feature.append(len(url))
        magic_feature.append(sum([url.count(shady_char) for shady_char in shady_chars]))
        magic_feature.append(sum([html.count(fake_char) for fake_char in fake_chars]))
        magic_feature.append(sum([html.count(ad_char) for ad_char in ad_chars]))
        magic_feature.append(sum([html.count(bg_char) for bg_char in bg_chars]))
        magic_features.append(magic_feature)
  
    magic_features = np.array(magic_features)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(magic_features)
    return scaler.transform(magic_features)

# Get article descrption from HTML
def get_description_from_html(html):
    soup = bs(html)
    description_tag = soup.find('meta', attrs={'name':'og:description'})
    description_tag = description_tag or soup.find('meta', attrs={'name':'description'})
    description_tag = description_tag or soup.find('meta', attrs={'property':'description'})
    if description_tag:
      description = description_tag.get('content') or ''
    else:
      description = ''
    return description

# Scrape description from website
def scrape_description(url):
    if not url.startswith('http'):
        url = 'http://' + url
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        description = get_description_from_html(html)
    except:
        description = 'the'

    return description

# Get description tokens using tokenizer
def prepare_description_data(data, tokenizer=None):
    text_data = []
  
    for data_point in tqdm_notebook(data):
        text_data.append(scrape_description(data_point[1]).strip().replace(u'\xa0', u' '))

    text_data = tokenizer.texts_to_sequences(text_data)
    text_data = pad_sequences(text_data, maxlen=20)
    return text_data

# Define final function to generate all training and validation features
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

# Create neural network model to classify articles as fake or real

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

# Fit model on 20 epochs and batch size 256
  
model = neural_model()
model.fit(x=[train_text_data, train_description_data, train_extension_data,\
             train_keyword_features, train_magic_features], y=train_targets,\
             validation_data=([val_text_data, val_description_data,\
                               val_extension_data, val_keyword_features, val_magic_features], val_targets),
             epochs=20, batch_size=256)

# Save model and tokenizer vocabulary for future use

model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("model.h5")

with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
