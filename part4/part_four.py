#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import json  
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[6]:


wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True)
genres = pd.read_json('genres.json.gz', orient='record', lines=True)
rotten = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True)

# omdb has the ====== imdb_id  and the plot is also there in it 
# rotten has the imdb_is and rotten_tomato_id
# genres has the wikidata_id
# wikidata has the imdb_id, wiki_id and the rotten_tomato_id



# In[7]:


movies = wikidata.merge(rotten, how='outer', on='imdb_id').merge(omdb, how='outer', on='imdb_id').merge(genres, how='outer', on='wikidata_id')
movies


# In[8]:


rotten.head()


# In[9]:


fig_pair=sns.pairplot(rotten) #used to check which columns are closely co-related.
fig_pair.savefig("output_paiplot.png")


# In[10]:


fig=sns.lmplot(x='critic_average',y='audience_average',data=rotten) #shows the linearity between audience and critic average ratting
fig.savefig("output.png")


# In[11]:


#sns.heatmap(movies.isnull(), cbar=False) #shows all the null values in the dataframe


# In[12]:


#filtering out the NaN and NA
movies = movies.dropna(subset=['omdb_awards'])
movies = movies[movies['omdb_awards'] != 'N/A']


# In[13]:


#seperating all the awards 
def awards(x):
    awards = re.findall(r'\d+',x) 
    awards = list(map(int, awards)) 
    total = np.sum(awards)
    return total
movies['Awards'] = movies['omdb_awards'].apply(awards)


# In[14]:


# filtering out the ratings
movies = movies.dropna(subset=['audience_average','critic_average']) # dropped all the null values
movies['critic_average'] = movies['critic_average']/2.0 # converted the rating out of 5


# In[ ]:





# In[15]:


#fig=sns.heatmap(movies.isnull(), cbar=False) # cross checking if there are still any null values in the ratings, after filtering the data
#fig.savefig("output.png")


# In[ ]:





# In[16]:


plt.scatter(movies['audience_average'],movies['Awards'])
plt.title('Audience average v/s Awards')
plt.xlabel('Audience average')
plt.ylabel('Awards and nominations')
plt.savefig('fig1.png')


# In[17]:


plt.scatter(movies['critic_average'],movies['Awards'])
plt.title('Critic average v/s Awards')
plt.xlabel('Critic average')
plt.ylabel('Awards and nominations')
plt.savefig('fig2.png')


# In[ ]:




