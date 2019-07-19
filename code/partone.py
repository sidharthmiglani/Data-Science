#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import json  
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[12]:


wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True)
genres = pd.read_json('genres.json.gz', orient='record', lines=True)
rotten = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True)
wikidata.drop(columns=['based_on','cast_member','director','made_profit','main_subject','series','filming_location','metacritic_id'])


# In[13]:


movies = wikidata.merge(rotten, how='outer', on='imdb_id').merge(omdb, how='outer', on='imdb_id').merge(genres, how='outer', on='wikidata_id')


# In[14]:


rotten.head()


# In[15]:


sns.pairplot(rotten) #used to check which columns are closely co-related.


# In[16]:


sns.lmplot(x='critic_average',y='audience_average',data=rotten) #shows the linearity between audience and critic average ratting


# In[17]:


sns.heatmap(movies.isnull(), cbar=False) #shows all the null values in the dataframe 


# In[18]:


#filtering out the NaN and NA
movies = movies.dropna(subset=['omdb_awards'])
movies = movies[movies['omdb_awards'] != 'N/A']


# In[22]:


#seperating all the awards from the string using regex
def awards_total(x):
    awards = re.findall(r'\d+',x) #regex find numbers
    awards = list(map(int, awards)) #to int datatype
    total_awards = np.sum(awards)
    return total_awards
movies['Awards'] = movies['omdb_awards'].apply(awards_total)


# In[24]:


# filtering out the ratings
movies = movies.dropna(subset=['audience_average','critic_average']) # dropped all the null values
#movies['critic_average'] = movies['critic_average']/2.0 # converted the rating out of 5


# In[ ]:





# In[26]:


sns.heatmap(movies.isnull(), cbar=False) # cross checking if there are still any null values in the ratings, after filtering the data


# In[28]:





# In[32]:


plt.scatter(movies['audience_average'],movies['Awards'])
plt.title('Audience average v/s Awards')
plt.xlabel('Audience average')
plt.ylabel('Awards and nominations')


# In[33]:


plt.scatter(movies['critic_average'],movies['Awards'])
plt.title('Critic average v/s Awards')
plt.xlabel('Critic average')
plt.ylabel('Awards and nominations')


# In[ ]:




