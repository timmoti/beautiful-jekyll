---
layout: post
published: false
title: 'Boardgames-O-Matic: Modeling for Predictions'
subtitle: >-
  Part 2 of 3 where I build a board games recommender system for
  Boardgamegeek.com users
date: 2017-11-5
bigimg: /img/BoardGameGeek.jpg
image: /img/bgg-logo.jpg
---
In part 1, we extracted the games and ratings data via webscraping and api calls to the Boardgamegeek website. In this section, we will be attempting to put these ratings through various models in order to generate recommendations for users on the website.

To recap, we have 120,679 users and 1807 games forming a N x M matrix of 120,679 x 1807 with a total of 7.7 million ratings. I have only selected users that have rated at least 10 games or more. Our matrix density is 3.54% or 96.45% sparse.

I will be utilizing both neighbourhood and latent factor models from the collaborative filtering approach to predict ratings for each user on the games he/she has yet to rate. For the neighbourhood method, I will be employing the use of the cosine similarity function. Matrix factorization will be accomplished via the SVD method as well as a Non-negative matrix factorization method utilizing weighted alternating least squares to minimize the loss function.

Offline evaluations will be conducted using RMSE.

# The Approach and Setup
In general, here's what we will be performing in order with each modeling technique:

1. Acquire train and test sets
2. Normalize train set ratings
3. Fit train set on model instance
4. Make predictions on test set
5. Determine RMSE
6. Provide recommendations for specified user

First we get our regular imports in along with some additional ones:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sqlite3
from scipy import sparse
from sklearn.metrics import mean_squared_error

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

sns.set_style('white')
```
We import scipy.sparse so that we can perform sparse matrix calculations on our sparse matrix which speeds up calculations. The sparse format stores only the necessary data in 3 arrays and ignores all the other empty cells.

We also import mean_squared_error to aid us in our calculation of the Root Mean Squared Error(RMSE) evaluation metric.

```python
df = pd.read_pickle('ratings_pickle')
df.shape
(120679, 1807)
```

We also import out pickled file from the part 1 and make sure it contains the matrix in the right shape

```python
games = pd.read_csv('bgg_gamelist.csv')
```

Let's not forget our gameslist as well which we will need to link with our ratings matrix through the gameids to identify the games recommended.

# Creating the base class

I decided to create a base Recommender class so that I am able to reuse some of the basic preprocessing methods with other modeling techniques down the pipeline. It helps to organize the code neatly too.

The Recommender class contains methods for preprocessing the data before modeling. The class takes in a ratings dataframe and allows one to create a sparse representation of it, train-test split it, normalize the training set, calculate mean by users and overall mean as well as obtaining the RMSE score for model evaluation.

```python
class Recommender(object):
    '''
    A class for building a recommender model
    
    Params
    ======
    ratings : (Dataframe)
        User x Item matrix with corresponding ratings
        
    '''
    
    def __init__(self, ratings):
        self.ratings = ratings
    
    def create_sparse(self, ratings):
        '''
        Creates sparse matrices by filling NaNs
        with zeros first
        '''
        ratings_sparse = sparse.csr_matrix(ratings.fillna(0))
        return ratings_sparse
        
    def train_test_split(self, ratings_sparse, test_size='default', train_size=None, random_state=None):
        '''
        Splits ratings data into train and test sets. 
        This implementation ensures that each test set
        will have at least one rating from each user.
        
        Params
        ======
        ratings_sparse : (Sparse matrix)
            A sparse representation of the original matrix
        
        test_size : (int)
            An integer indicating number of random ratings 
            from each user to be allocated to the test set
        
        train_size : (int)
            An integer indicating number of random ratings
            from each user to be allocated to the train set
            
        random_state : (int)
            Random seed for reproducibility
        '''
        r = np.random.RandomState(random_state)
        
        if test_size == 'default':
            test_size = None
        if test_size is None and train_size is None:
            test_size = 1
        if test_size == float or train_size == float:
            raise ValueError('test_size or train_size must be a positive integer')
        if train_size is not None: 
            test_size = None
        
        if test_size is not None:
            train_set = self.ratings.copy().as_matrix()
            test_set = np.full(self.ratings.shape, np.nan)

            for idx, row in enumerate(ratings_sparse):
                #Get column ids based on random choice of available games in current user
                test_indices = r.choice(row.indices, test_size, replace=False)
                test_data = []
                for val in test_indices:
                    index = list(row.indices).index(val)
                    test_data.append(row.data[index])
                train_set[idx, test_indices] = np.nan
                test_set[idx, test_indices] = test_data
            train_set = pd.DataFrame(train_set, index=self.ratings.index, columns=self.ratings.columns)
            test_set = pd.DataFrame(test_set, index=self.ratings.index, columns=self.ratings.columns)
                
        else:
            test_set = self.ratings.as_matrix()
            train_set = np.full(self.ratings.shape, np.nan)
            
            for idx, row in enumerate(ratings_sparse):
                train_indices = r.choice(row.indices, train_size, replace=False)
                train_data = []
                for val in train_indices:
                    index = list(row.indices).index(val)
                    train_data.append(row.data[index])
                test_set[idx, train_indices] = np.nan
                train_set[idx, train_indices] = train_data
            train_set = pd.DataFrame(train_set, index=self.ratings.index, columns=self.ratings.columns)
            test_set = pd.DataFrame(test_set, index=self.ratings.index, columns=self.ratings.columns)
        
        assert self.ratings.notnull().sum().sum() == train_set.notnull().sum().sum() + test_set.notnull().sum().sum()
    
        return train_set, test_set
        
    
    def normalize(self, ratings_sparse):
        '''Normalizes each user by their mean. Accepts a sparse array'''
        ratings = np.zeros_like(self.ratings)
        for idx, i in enumerate(ratings_sparse):
            ratings[idx, i.indices] = i.data-i.data.mean()    
        return ratings
    
    def get_user_mean(self):
        '''Calculate mean rating values for each user'''
        mean = np.mean(self.ratings, axis=1)
        return mean
    
    def get_overall_mean(self, mean):
        '''
        Outputs mean of entire dataframe
        
        Params
        ======
        
        mean : (pd.Series or nd_array)
            Vector of user means across their ratings
        '''
        mean_all = np.mean(mean)
        return mean_all
    
    def get_baseline_rmse(self, test, mean_all):
        '''
        Ouputs the baseline rmse by comparing test 
        set ratings with the mean rating for each user
        
        Params
        ======
        
        test : (nd_array)
            Numpy array of test results
            
        test_sparse : (Sparse matrix)
            Sparse representation of test matrix
        
        mean_all : (float)
            Value of mean of entire ratings matrix
        '''
        actual = test[test.nonzero()].flatten()
        preds = [mean_all for val in test[test.nonzero()].flatten()]
        prediction = preds
        return np.sqrt(mean_squared_error(actual, prediction))
    
    def get_rmse(self, test, prediction):
        '''
        Returns the rmse score from a prediction result
        
        Params
        ======
        
        test : (nd_array
            Numpy array of test results
            
        prediction (nd_array)
            Predicted scores after training through model
        '''
        actual = test[test.nonzero()].flatten()
        prediction = prediction[prediction.nonzero()].flatten()
        return np.sqrt(mean_squared_error(actual, prediction))

```
The most interesting method in the base class would probably be the train_test_split method as I had to create one to ensure that a certain number of ratings would be in the test set for each user as opposed to a random assignment. In my implementation, you are able to enter the number of ratings you want in the test set or the training set. It is possible to set a low training set value to simulate the cold start problem using this function.

# Neighbourhood method - Cosine similarity recommender
Our first method to determine recommendations is by calculating the similarity between users and items. I have decided to employ the cosine similarity function owing to its favourable representation in the recommender systems community.

A similarity matrix is first obtained between each item pair or user pair. A prediction function is then used to predict ratings for results held out in the test set in order to compare the accuracy through the RMSE loss function.

In this case, we will only be utilizing item-item similarities as our neighborhood method of choice due to less intensive computation required.

A subclass was built off the base Recommender class so as to be able to utilize some of its functions like calculating RMSE which will be used for all our models.

```python
from sklearn.metrics.pairwise import cosine_similarity 

class CosineSim(Recommender):
    '''
    A Recommender class that uses the cosine similarity 
    function to recommend games
    
    Params
    ======
    ratings : (Dataframe)
        Ratings matrix
    '''
    def __init__(self, ratings):
        self.ratings = ratings
    
    def find_similarity(self, ratings_sparse, kind='item'):
        '''
        Finds the cosine similarity in either 
        item-item or user-user configurations
        of a sparse matrix
        
        Params
        ======
        ratings_sparse : (Sparse matrix)
            Sparse representation of ratings matrix
        
        kind : (string)
            Default is item for item-item similarity. 
            Other option is user for user-user similarity
        '''
        if kind == 'item':
            similarities = cosine_similarity(ratings_sparse.T)
        elif kind == 'user':
            similarities = cosine_similarity(ratings_sparse)
        return similarities
    
    def predict_for_test(self, train, test_sparse, similarity, kind='item'):
        '''
        Predicts the scores for games in test set
        
        Params
        ======
        train : (Dataframe)
            Dataframe of training set
            
        test_sparse : (Sparse matrix)
            Sparse representation of test set
            
        similarity : (nd_array)
            Similarity matrix of user-user or item-item pairs
            
        kind : (string)
            Default is 'item' for item-item predictions.
            Alternative is 'user' for user-user predictions
            
        '''
        prediction = np.zeros_like(train)

        if kind == 'item':
            train = train.fillna(0).as_matrix()
            for row, val in enumerate(test_sparse):
                for col in val.indices:
                    sim = similarity[col]
                    rated = train[row].nonzero()
                    prediction[row, col] = np.sum(sim[rated]*train[row][rated])/np.sum(np.abs(sim[rated]))

        elif kind == 'user':
            mean  = train.apply(np.mean, axis=1)
            std = train.apply(np.std, axis=1)

            for row, val in enumerate(test_sparse):
                avg_rating = mean.iloc[row]
                sim = similarity[row]
                ind_std = std.iloc[row]

                for col in val.indices:
                    prediction[row, col] = avg_rating + np.sum(sim*train.iloc[:, col])/np.sum(np.abs(sim))

        return prediction    
        
    def recommend(self, user, similarity, games, kind='item'):
        '''
        Recommends list of games to specified user
        
        Params
        ======
        user : (string)
            username with at least 10 ratings in database
            
        similarity : (nd_array)
            Similarity matrix of user-user or item-item pairs
            
        games : (Dataframe)
            Dataframe of game list
        
        kind : ('string)
            Default is 'item' for item-item predictions.
            Alternative is 'user' for user-user predictions
        '''
        preds = np.zeros_like(self.ratings.loc[user])

        if kind == 'item':
            ratings_array = self.ratings.fillna(0).as_matrix()
            person = ratings_array[self.ratings.index.get_loc(user)]
            rated = person.nonzero()[0]
            for idx in range(ratings_array.shape[1]):
                if idx not in rated:
                    sim = similarity[idx]
                    preds[idx] = np.sum(sim[rated]*person[rated])/np.sum(np.abs(sim[rated]))

        elif kind == 'user':

            rated = self.ratings.loc[user].dropna().index
            std = np.std(self.ratings.loc[user])
            mean  = np.mean(self.ratings.loc[user])
            sim = similarity[self.ratings.index.get_loc(user)]

            for idx, col in enumerate(ratings.columns):
                if col not in rated:
                    preds[idx] = mean + np.sum((sim*ratings[col]))/np.sum(np.abs(sim))

        predictions = pd.Series(preds, index=self.ratings.columns, name='predictions')
        recommendations = games.join(predictions, on='gameid')
        return recommendations.sort_values('predictions', ascending=False)
```
I have allocated the ability to select user-user or item-item calculations for the find_similarity, predict_for_test and recommend functions which works on a smaller toy dataset. It's interesting to note that the prediction function I coded for thie Cosine Similarity model predicts only for the holdout values on the test set and not on the entire matrix for the sake of computational speed.

Following our general approach, here are the steps to model, predict and generate recommendations utilizing the Cosine Similarity function:

## Preprocess data
```python
#Creating an instance of the Recommnder class
rec = Recommender(df)

#Creating a sparse matrix of the full data
df_sparse = rec.create_sparse(df)

#Performing train-test split 
%time train, test = rec.train_test_split(df_sparse, test_size=2, random_state=123)

#Create sparse matrix for training set
train_sparse = rec.create_sparse(train)

#Create sparse matrix for test set
test_sparse = rec.create_sparse(test)
```

## Modeling - Form Similarity Matrix
```python
#Create an instance of the Cosine Similarity Recommender class
cossim = CosineSim(df)
```
The first thing we should normally do is normalize the training set to account for variances in how users rate the games. We do not normalize the training set here as the cosine similarity function from sklearn already does it for us.

```python
#Acquiring item-item similarity matrix
%time item_sims = cossim.find_similarity(train_sparse)

CPU times: user 5.83 s, sys: 357 ms, total: 6.19 s
Wall time: 6.3 s
```
## Predicting scores on test set
```python
#Predicting the scores for test set
%time item_preds = cossim.predict_for_test(train, test_sparse, item_sims)

CPU times: user 36 s, sys: 5.92 s, total: 41.9 s
Wall time: 43.5 s
```
We measure wall time so we can appreciate how long it takes to generate these recommendations which is important when we decide to push it to production.

## Evaluate with RMSE
First we calculate our baseline RMSE which is taken as the same value of the mean rating for the entire dataset predicted for each datapoint in the test set.

```python
#Converting test set to an array
y = test.fillna(0).values

#Getting user means
mean = rec.get_user_mean()

#Calculating overall mean
overall_mean = rec.get_overall_mean(mean)
overall_mean

7.398810764577983

#Computing baseline error
baseline_error = rec.get_baseline_rmse(y, overall_mean)
baseline_error

1.4869803145678728
```

We then calculate our RMSE score for the Cosine Similarity model.

```python
#Calculating the error
error_cos = cossim.get_rmse(y, item_preds)
error_cos

1.3339479358142938
```
We can observe a decrease of about 10% from the baseline score.

## Performing recommendations

We will be performing recommendations for myself as I am an active user of the site and have rated quite a number of games (226 to be exact). We calculate the similarity matrix again, this time utilizing the entire dataset before making the predictions and utilizing the recommendation function

```python
#Calculate item-item similarity matrix of entire ratings data
%time all_sims = cossim.find_similarity(df_sparse)





