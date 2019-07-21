#!/usr/bin/env python
# coding: utf-8

# In[24]:


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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report





# In[25]:


# omdb has the ====== imdb_id  and the plot is also there in it 
# rotten has the imdb_is and rotten_tomato_id
# genres has the wikidata_id
# wikidata has the imdb_id, wiki_id and the rotten_tomato_id


wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True)
genres = pd.read_json('genres.json.gz', orient='record', lines=True)
rotten = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True)


# In[26]:


tokens=omdb['omdb_plot']


# In[27]:


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


# In[28]:


tokens=tokens.apply(text_process)


# In[29]:


omdb['tokenized']=tokens


# In[30]:


omdb['Polarity_full_string'] = omdb.omdb_plot.apply(lambda x: TextBlob(x).sentiment.polarity)


# In[31]:


def summary(x):
    
    blob = TextBlob(x)
    nouns = list()
    summary_list = list()
    for word, tag in blob.tags:
        if tag == 'NN':
            nouns.append(word.lemmatize())

    for item in random.sample(nouns, 3):
        word = Word(item).pluralize()
        summary_list.append(word)
        
    
    return  ' '.join(summary_list)


# In[32]:


# test code  # omdb has 9676 rows
#blob = TextBlob(omdb.omdb_plot[5])
#nouns = list()
#summary_list = list()
#for word, tag in blob.tags:
#    if tag == 'NN':
#        nouns.append(word.lemmatize())

#print ("This text is about...")
#for item in random.sample(nouns, 5):
#    word = Word(item).pluralize()
#    summary_list.append(word)
    
    
#print(' '.join(summary_list))    


# In[33]:


#compressed=omdb['omdb_plot']
#compressed=compressed.apply(summary)


# In[34]:


summary(omdb.omdb_plot[4])


# In[ ]:





# In[ ]:





# In[50]:


omdb.Polarity_full_string.hist()
movies = rotten.merge(omdb, how='inner', on='imdb_id')
movies=movies[['imdb_id','rotten_tomatoes_id','omdb_plot','Polarity_full_string','audience_average','tokenized']]
movies[movies['Polarity_full_string']>0.5].to_csv('positive_movies.csv')
movies = movies.dropna(subset=['audience_average'])
movies = movies[movies['audience_average'] != 'N/A']
movies['audience_average']=movies['audience_average'].astype(int)


# In[36]:


# lets create the bag of words
bow_transformer = CountVectorizer(analyzer=text_process).fit(omdb.tokenized)


# In[37]:


# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[38]:


#lets use.transform on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages
plot_bow = bow_transformer.transform(omdb.tokenized)


# In[39]:


print('Shape of Sparse Matrix: ', plot_bow.shape)
print('Amount of Non-Zero occurences: ', plot_bow.nnz)


# In[40]:


# since, the counting is done now the term weighting and normalization can be done with TF-IDF
tfidf_transformer = TfidfTransformer().fit(plot_bow)


# In[41]:


#To transform the entire bag-of-words corpus into TF-IDF
plot_tfidf = tfidf_transformer.transform(plot_bow)
print(plot_tfidf.shape)


# In[42]:


# now let's train the model
plot_train, plot_test, rating_train, rating_test =train_test_split(movies['tokenized'], movies['audience_average'], test_size=0.3)


# In[43]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[44]:


pipeline.fit(plot_train,rating_train)


# In[45]:


predictions = pipeline.predict(plot_test)


# In[48]:


report=classification_report(predictions,rating_test,output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('classification_report.csv')


classification_report(predictions,rating_test,output_dict=True)


# In[ ]:





# In[ ]:




