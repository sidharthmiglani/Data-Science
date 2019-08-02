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
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import collections
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

print("RUNNING PROGRAM:")
'''
wikidata = pd.read_json('data/wikidata-movies.json.gz', orient='record', lines=True)
genres = pd.read_json('data/genres.json.gz', orient='record', lines=True)
rotten = pd.read_json('data/rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_json('data/omdb-data.json.gz', orient='record', lines=True)
# omdb has the ====== imdb_id  and the plot is also there in it
# rotten has the imdb_id and rotten_tomato_id along with all the ratings
# genres has the wikidata_id
# wikidata has the imdb_id, wiki_id and the rotten_tomato_id
tokens=omdb['omdb_plot']
'''

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
    nonstop=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return  ' '.join(nonstop)
'''
tokens=tokens.apply(text_process)

omdb['tokenized']=tokens

omdb['polarity_full_string'] = omdb.omdb_plot.apply(lambda x: TextBlob(x).sentiment.polarity)

len(omdb.tokenized)

#write to to_csv ombd
omdb.to_csv('omdb_clean.csv')
'''
#upload csv
wikidata = pd.read_json('data/wikidata-movies.json.gz', orient='record', lines=True)
rotten = pd.read_json('data/rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_csv('omdb_clean.csv')

print("DONE")
def give_sentiment(x):
    if(x >= 0.5):
        return 'Really Positive'
    elif(x <0.5 and x>=0):
        return 'Positive'
    elif((x<0) and (x>= -0.5)):
        return 'negative'
    elif((x<-0.5) and (x>=-1)):
        return 'Really negative'

    # elif(x <0 and x>= -0.25):
    #     return 'Positive'
    # elif((x<-0.25) and (x>= -0.5)):
    #     return 'negative'
    # elif((x<-0.5) and (x>= -0.75)):
    #     return 'Really negative'
    # elif((x<-0.75) and (x>= -1)):
    #     return 'Really negative'

omdb['movie_nature'] = (omdb.polarity_full_string.apply(lambda x: give_sentiment(x)))

movies = rotten.merge(omdb, how='inner', on='imdb_id')

sns.heatmap(movies.isnull(), cbar=False)

movies = movies.dropna(subset=['audience_average','critic_average'])
movies = movies[movies['omdb_awards'] != 'N/A']


# let's clean wikiData
# dropped all the null and NAN values
wikidata = wikidata.dropna(subset=['cast_member'])
wikidata = wikidata[wikidata['cast_member'] != 'N/A']

movies = movies.merge(wikidata,how='inner', on='imdb_id')

# extracting the total number of cast members for each movies
def cast_members_length(x):
    return len(x)
wikidata['cast_length'] = (wikidata.cast_member.apply(lambda x: cast_members_length(x)))

# deleting all the movies whose ast members are lesser than 3
new_wikidata=wikidata[wikidata['cast_length'] >= 3 ]

# function to extract the first three cast members
def cast_member_one(x):
    return (x[0])[1:]
def cast_member_two(x):
    return (x[1])[1:]
def cast_member_three(x):
    return (x[2])[1:]

# putting the cast members in three respective columns
new_wikidata['cast_one'] = (new_wikidata.cast_member.apply(lambda x: cast_member_one(x)))
new_wikidata['cast_two'] = (new_wikidata.cast_member.apply(lambda x: cast_member_two(x)))
new_wikidata['cast_three'] = (new_wikidata.cast_member.apply(lambda x: cast_member_three(x)))
new_wikidata.to_csv('deleteme2.csv')
# print(new_wikidata)

# let's merge the cast members data into movies dataframe
movies = movies.merge(new_wikidata,how='inner', on='imdb_id')


# machine learning model
model = make_pipeline(StandardScaler(),SVC(kernel='rbf'))

# print(movies)

movies = movies.drop(movies.index[3])
movies.to_csv ('deleteme.csv', index = None, header=True)
x = movies[['critic_average','cast_one','cast_two','cast_three']]
y = movies.movie_nature

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)

model.fit(X_train,y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
