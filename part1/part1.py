import sys
import os
from pyspark.sql import SparkSession, functions, types
import sys
from math import radians, cos, sin, asin, sqrt
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import datetime as dt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pykalman import KalmanFilter
from xml.dom.minidom import parse, parseString
from pandas import DataFrame
import difflib
import gzip
import json
from scipy import stats
from pylab import *
import re
import datetime


'''
do people like the first movie better than the second one in
a series
'''
'''
how to run:
export PYSPARK_PYTHON=python3

export PATH=$PATH:/home/sarbjot/spark-2.4.3-bin-hadoop2.7/bin

export JAVA_HOME=/usr/lib/jvm/java-8-oracle/

spark-submit part1.py wiki-copy-1 output_statsQ
'''
spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

schema = types.StructType([
    types.StructField('wikidata_id', types.StringType()),
    types.StructField('label', types.StringType()),
    types.StructField('imdb_id', types.StringType()),
    types.StructField('rotten_tomatoes_id', types.StringType()),
    types.StructField('metacritic_id', types.StringType()),
    types.StructField('enwiki_title', types.StringType()),
    types.StructField('genre', types.StringType()),
    types.StructField('main_subject', types.StringType()),
    types.StructField('filming_location', types.StringType()),
    types.StructField('director', types.StringType()),
    types.StructField('cast_member', types.StringType()),
    types.StructField('series', types.StringType()),
    types.StructField('publication_date', types.StringType()),
    types.StructField('based_on', types.StringType()),
    types.StructField('country_of_origin', types.StringType()),
    types.StructField('original_language', types.StringType()),
    types.StructField('made_profit', types.StringType()),

])


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    # 1. read the input direcotry of .csv.gz files
    #weather = spark.read.csv(in_directory, schema=observation_schema)
    wiki_df = spark.read.json(in_directory, schema=schema)
    wiki_df = wiki_df.select(wiki_df['label'],wiki_df['rotten_tomatoes_id'],wiki_df['enwiki_title'],wiki_df['series'], wiki_df['publication_date']  )
    #wiki_df.show()

    # filter movies not in series

    wiki_df = wiki_df.filter(wiki_df.series. isNotNull())
    #wiki_df.show()

    #write to json ( only one json output file!!!!)
    wiki_df.coalesce(1).write.format('json').save('cleaned_data',mode= 'overwrite')

    #read from json in pandas dataframe (easier to work with since series data is smaller)
    print(os.listdir(path='cleaned_data'))
    folder_list = os.listdir(path='cleaned_data')

    #find the right json file in the folder
    r = re.compile(r'json$')
    newlist = list(filter(r.search, folder_list)) # Read Note
    print(newlist)
    #print(the_json_file)
    series_df = pd.read_json('cleaned_data/'+newlist[0], lines=True)


    grouped_series  = series_df.groupby('series')
    #grouped_series = grouped_series.filter(lambda x: x['label'].count()<=1)
    print(grouped_series.first())

    #convert publication date into datetime objects
    compare_ratings_df = pd.DataFrame(columns= ['movie_name1', 'rotten_id1','rotten_rating1','movie_name2', 'rotten_id2','rotten_rating2'])
    for serie, serie_df in grouped_series:
        #print(serie)
        serie_df['publication_data'] = pd.to_datetime(serie_df['publication_date'],format='%Y-%m-%d')
        #print(series_df.select_dtypes(include=[np.datetime64]))
        serie_df = serie_df.sort_values(by='publication_date')
        #print(serie_df)
        #take each group. take first and second. insert into new dataframe
        if serie_df['label'].count() >=2 :
            # find oldest and second oldest and put ratings in dataframe
            #make np array of all the releveant data
            np_array = np.array([serie_df.iloc[0]['enwiki_title'], serie_df.iloc[0]['rotten_tomatoes_id'],0,serie_df.iloc[1]['enwiki_title'], serie_df.iloc[1]['rotten_tomatoes_id'],0])
            #convert to series
            series_instance = pd.Series(np_array, index=['movie_name1', 'rotten_id1','rotten_rating1','movie_name2', 'rotten_id2','rotten_rating2'])
            # add the series row to the dataframe
            compare_ratings_df = compare_ratings_df.append(series_instance,ignore_index=True)



    #find ratings assocated to each movie
    rotten_df = pd.read_json('wiki-copy-1/rotten-tomatoes.json', lines=True)
    rotten_df = rotten_df.drop(['audience_average', 'audience_ratings','critic_average','critic_percent','imdb_id'], axis=1)
    #create series for ratings1 and ratings2. keeping appending it. then insert into DataFrame
    ratings1_np = np.array([])
    ratings2_np = np.array([])

    for index,row in compare_ratings_df.iterrows():
        #for first movie
        id1 = row[1]
        rotten_row = rotten_df.loc[rotten_df['rotten_tomatoes_id'] == id1]
        a_rating1 = rotten_row['audience_percent']
        ratings1_np = np.append(ratings1_np, a_rating1, axis=0)

        #for second movie
        id2 = row[4]
        rotten_row2 = rotten_df.loc[rotten_df['rotten_tomatoes_id'] == id2]
        a_rating2 = rotten_row2['audience_percent']
        ratings2_np = np.append(ratings2_np, a_rating2, axis=0)



    compare_ratings_df['rotten_rating1'] = pd.Series(ratings1_np)
    compare_ratings_df['rotten_rating2'] = pd.Series(ratings2_np)

    compare_ratings_df.dropna(inplace=True, axis='rows')

    #print(compare_ratings_df)
    compare_ratings_df.to_csv (r'readyForAnalysis.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

    '''
    # Attempt to Normalize the data

    print("normal test rotten_rating1")
    print(stats.normaltest(compare_ratings_df['rotten_rating1']).pvalue)
    compare_ratings_df['rotten_rating1']= compare_ratings_df['rotten_rating1']**2
    fig, ax = plt.subplots()

    ax.hist(compare_ratings_df['rotten_rating1'], 30, density=4)

    plt.show()
    '''
    print("\n\n\n\n\n\n\n\nAnalysis:\n")
    print("\nLevene's Test")
    print("H0 of Levene's Test: two samples have equal variance. If p>0.05 proceed as if equal variance\np-value is:\n")
    print(stats.levene(compare_ratings_df['rotten_rating1'], compare_ratings_df['rotten_rating2']).pvalue)
    print("The two data sets have close enough variance for t-test\n\n\n")


    print("Normal Test: if p > 0.05 then proceed as if normal")
    print("p value of normal test on rotten_rating1\np-value is:")
    print(stats.normaltest(compare_ratings_df['rotten_rating1']).pvalue)
    print("p value of normal test on rotten_rating2\np-value is:")
    print(stats.normaltest(compare_ratings_df['rotten_rating2']).pvalue)

    print("Data not normal, therefore use Mann-Whitney U test:\n")
    print("\nNull hypothesis = first movie in series recieved lower than or equal ratings to second movie in series, \n alternative hypothesis: first movie recieved ratings greater than second movie in series")
    print("p-value of Mann-Whitney U test:")
    print(stats.mannwhitneyu(compare_ratings_df['rotten_rating1'], compare_ratings_df['rotten_rating2'], alternative='greater').pvalue)
    print("P-value is less than 0.05 level of signifigance, therefore reject the null and accetp the alternative hypothesis")
    # VISUALIZTION
    legend = ['First Movie', 'Sequel']
    first_movie = compare_ratings_df['rotten_rating1']
    sequel = compare_ratings_df['rotten_rating2']
    plt.hist([first_movie, sequel], color=['blue', 'red'])
    plt.xlabel("Rating Scores")
    plt.ylabel("Frequency")
    plt.legend(legend)
    plt.xticks(np.arange(0, 110, step=10))
    plt.yticks(np.arange(0, 36, step=2))
    plt.title('Part 1:\n Ratings of First Movie vs Second Movie in Series')
    #plt.show()
    plt.savefig('firstmovie_vs_sequel.png')


if __name__=='__main__':
    main()
