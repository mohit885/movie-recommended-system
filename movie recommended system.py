#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies


# In[5]:


movies.shape


# In[6]:



movies.columns


# In[7]:


credits


# In[8]:


credits.shape


# In[9]:


credits.columns


# In[10]:


credits=  credits.rename(columns = {'movie_id':'id'})
tmdb = movies.merge(credits,on='id')
tmdb


# In[16]:


tmdb.to_csv('tmdb_tab',index=False)
read_file = pd.read_csv('tmdb_tab')
read_file.to_excel(r'C:\Users\Deepak\Desktop\tmdb_tab.xlsx', index = None, header=True)


# In[11]:


tmdb.head().transpose()


# In[12]:


tmdb.shape


# In[ ]:


tmdb.info()


# In[ ]:


tmdb.isna().sum()


# In[ ]:


# release years of the movies
tmdb['release_year']= pd.DatetimeIndex(tmdb['release_date']).year


# In[ ]:


# most of the movies release in year
mrl_yr= tmdb.groupby('release_year')['id'].count().sort_values(ascending=False)
mrl_yr.head(10)


# In[ ]:





# In[ ]:


# most of the movies has which original language
tmdb['original_language'].nunique()
orla= tmdb.groupby('original_language')['id'].count().sort_values(ascending=False)
orla


# # different spoken languages
# 

# In[ ]:


import ast

def extract_language(language_list):
    language = []
    language_list = ast.literal_eval(language_list)
    for language_dict in language_list:
        language.append(language_dict["name"])

    return language


# In[ ]:


lang = tmdb['spoken_languages'].apply(lambda x: extract_language(x))
lang_types = lang.explode()
lang_types.nunique()
#lang_types.unique()
diflan= lang_types.value_counts().sort_values(ascending= False)
diflan


# # production countries

# In[ ]:


import ast

def extract_country(country_list):
    country = []
    country_list = ast.literal_eval(country_list)
    for country_dict in keywords_list:
        country.append(country_dict["name"])

    return country


# In[ ]:


country = tmdb['production_countries'].apply(lambda x: extract_keywords(x))
country_types = country.explode()
country_types.nunique()


# In[ ]:


cou= country_types.value_counts().sort_values(ascending= False)
cou


# In[ ]:


tmdb= tmdb.drop(['homepage','production_countries','release_date','status','tagline','title_x','title_y'],axis=1)


# In[ ]:


tmdb


# Expensive movies

# In[ ]:


budget_sort = tmdb.sort_values('budget',ascending=False)
plt.figure(figsize=(12,8))
sns.barplot(x = budget_sort.budget.head(10),y = budget_sort.original_title.head(10))
plt.xlabel('Budget cost in million dollars')
plt.ylabel('movie names')
plt.title('Top 10 Expensive Films')


# Top 10 popular movies.

# In[ ]:


popular_movie = tmdb.sort_values('popularity',ascending=False)
plt.figure(figsize=(12,8))
sns.barplot(x = popular_movie['popularity'].head(10),y=popular_movie['original_title'].head(10))
plt.xlabel('Popularity')
plt.ylabel('Movies')
plt.title('Top 10 Popular Movies')


# movies which given major profit
# 

# In[ ]:


tmdb['profit'] = (tmdb['revenue']-tmdb['budget'])
profitable_movie= tmdb.sort_values('profit',ascending= False)
plt.figure(figsize=(12,8))
sns.barplot(x = profitable_movie['profit'].head(10),y=profitable_movie['original_title'].head(10))
plt.xlabel('profits')
plt.ylabel('Movies')
plt.title('Top 10 Profitable Movies')


# longest running movies
# 

# In[ ]:


runtime_movie = tmdb.sort_values('runtime',ascending=False)
runtime_movie['runtime'] = runtime_movie.runtime.apply(lambda x: x/60)

plt.figure(figsize=(12,8))
sns.barplot(x = runtime_movie['runtime'].head(10),y=runtime_movie['original_title'].head(10))
plt.xlabel('Runtime in hours',weight='bold')
plt.ylabel('Name of the movie',weight='bold')
plt.title('Top 10 Movies with Highest Runtimes',weight='bold')


# Most occurance genre

# In[ ]:


tmdb['genres'][21]


# In[ ]:



import ast

def extract_genres(genre_list):
    genre = []
    genre_list = ast.literal_eval(genre_list)
    for genre_dict in genre_list:
        genre.append(genre_dict["name"])

    return genre


# In[ ]:


tmdb['new_genres'] = tmdb['genres'].apply(lambda x: extract_genres(x))


# In[ ]:


print(tmdb['genres'][0])
print(tmdb['new_genres'][0])


# In[ ]:


genres_types = tmdb['new_genres'].explode()
genres_types.nunique()
genres_types.unique()


# In[ ]:


genres_types.value_counts().sort_values().plot(kind='barh')


# In[ ]:


import ast

def extract_production(production_list):
    production = []
    production_list = ast.literal_eval(production_list)
    for production_dict in production_list:
        production.append(production_dict["name"])

    return production


# In[ ]:


tmdb['new_pdhouse'] = tmdb['production_companies'].apply(lambda x: extract_production(x))


# In[ ]:


tmdb['new_pdhouse']


# In[ ]:


pdhouse_types = tmdb['new_pdhouse'].explode()
pdhouse_types.nunique()
#pdhouse_types.unique()


# # weighted average of movies
# 

# In[ ]:





# In[ ]:


v=tmdb['vote_count']
R=tmdb['vote_average']
C=tmdb['vote_average'].mean()
m=tmdb['vote_count'].quantile(0.70)


# In[ ]:


tmdb['weighted_average']=((R*v)+ (C*m))/(v+m)


# In[ ]:


tmdb['weighted_average']


# In[ ]:


tmdb_1= tmdb.sort_values('weighted_average',ascending=False)[['original_title', 
                                                              'vote_count', 'vote_average', 
                                                              'weighted_average', 'popularity']]


# In[ ]:





# In[ ]:


# we are getting the most recommended movies on the basis of weighted_average
tmdb_1.head(10)


# In[ ]:


# but we haven't consider the popularity can a valid point in the recommendation


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaling=MinMaxScaler()
x= tmdb_1[['weighted_average','popularity']]
scaled_data= scaling.fit_transform(x)
scaled_data= pd.DataFrame(scaled_data,columns= x.columns)


# In[ ]:


tmdb_1[['normalized_weight_average','normalized_popularity']]= scaled_data
tmdb_1['original_title']= tmdb['original_title']


# In[ ]:


# we are giving the priortiy of 70% on weighted average and 30% priority to popularity
tmdb_1['score'] = tmdb_1['normalized_weight_average'] * 0.70 + tmdb_1['normalized_popularity'] * 0.30


# In[ ]:


tmdb_1.sort_values(['score'], ascending=False)[['original_title', 
                                                'normalized_weight_average', 'normalized_popularity', 'score']]


# In[ ]:


sns.barplot(x= tmdb_1.sort_values(['score'],ascending=False)['score'].head(10), 
            y=tmdb.sort_values(['score'], ascending=False)['original_title'].head(10))
plt.xlabel('score')
plt.ylabel('movies')
plt.title('best rate and popular movie')


# # content based filtering

# In[ ]:


tmdb.head(10)


# In[ ]:


#Import TfIdfVectorizer from scikit-learn
#for vectorizing the text
from sklearn.feature_extraction.text import TfidfVectorizer
#extracting the stop wards
tfidf = TfidfVectorizer(stop_words='english')
# filing the null spaces
tmdb['overview'] = tmdb['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(tmdb['overview'])
tfidf_matrix.shape


# In[ ]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel
# Computing  the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[ ]:


indices = pd.Series(tmdb.index, index=tmdb['original_title']).drop_duplicates()


# In[ ]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return tmdb['original_title'].iloc[movie_indices]


# In[ ]:


get_recommendations('Avatar')

