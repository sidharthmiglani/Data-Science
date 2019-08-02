In your repository's README (or README.md) file, you should document your code and how to run it: required libraries, commands (and arguments), order of execution, files produced/expected. You should do this because (1) you should always do that, and (2) to give us some hope of running your code.



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


## Authors

* **Sarbjot Singh** - *Initial work* - [PurpleBooth](https://github.com/sarbjot-14)
