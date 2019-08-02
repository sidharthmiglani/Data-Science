
# ProjectMovies
Final Project in Computational Data Science. Answer questions using statistical analysis and machine learning on movie data.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Prerequisites

 * Python 3
 * Spark

###  Part1

#### This part answers if the first movie in the a series is better than   the second using a t-test

### Files included
  1. wiki_copy-1 : data about movies and ratings1
  2. cleaned_data : spark cleans, data from wiki_copy_1 and outputs for next step
  3. part1.py : program that does all computatoins
  4. readyForAnalysis : produced for user to see data the t-test is performed on

    1. move to part1 folder  
    2. set up enviroment
      ```
      export PATH=$PATH:/home/<path to bin>
      ```
      ```
      export JAVA_HOME=/<path to java module>
      ```
      ```
      spark-submit part1.py wiki-copy-1 output_statsQ
      ```
    3. See output in terminal
   
###  Part2

#### This machine learning part predicts the polarity of the movies based on cast members and ratings of the movie. 

### Files included

  1. data : Downloaded data from sfu cluster
  2. omdb_clean.csv : Cleaned omdb data stored as a csv
  3. ml1.py : program that does all computatoins
  
  
    1. In your Terminal: move to part2 folder  
    2. install textblob, nltk and any other library that is not already installed already.
    3. Run python3 ml1.py (If if asks you to install any other packages then usually do pip install *pkg name*. Hopefully will work)
    3. See output in terminal

## Authors
* **Sarbjot Singh** - (https://github.com/sarbjot-14)
* **Sidharth Miglani** - (https://github.com/sidharthmiglani)
* **Ronit Chawla** - (https://github.com/ronitchawla)
