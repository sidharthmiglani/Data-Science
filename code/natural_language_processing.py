#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import json  
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from textblob import Word
import random



# In[2]:


wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True)
genres = pd.read_json('genres.json.gz', orient='record', lines=True)
rotten = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True)


# In[3]:


tokens=omdb['omdb_plot']


# In[5]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[6]:


tokens=tokens.apply(text_process)


# In[13]:


omdb['tokenized']=tokens


# In[19]:


omdb['Polarity_full_string'] = omdb.omdb_plot.apply(lambda x: TextBlob(x).sentiment.polarity)


# In[71]:


def summary(x):
    #random.choice(x)
    blob = TextBlob(x)
    nouns = list()
    summary_list = list()
    for word, tag in blob.tags:
        if tag == 'NN':
            nouns.append(word.lemmatize())

    for item in random.sample(nouns, 7):
        word = Word(item).pluralize()
        summary_list.append(word)
        
    return summary_list


# In[72]:


# test code 
blob = TextBlob(omdb.omdb_plot[0])
nouns = list()
summary_list = list()
for word, tag in blob.tags:
    if tag == 'NN':
        nouns.append(word.lemmatize())

print ("This text is about...")
for item in random.sample(nouns, 7):
    word = Word(item).pluralize()
    summary_list.append(word)
    print (word)
    
print(summary_list)    


# In[75]:


#omdb['plot_summary'] = omdb['omdb_plot'].apply(summary)


# In[74]:


summary(omdb.omdb_plot[0])


# In[ ]:




