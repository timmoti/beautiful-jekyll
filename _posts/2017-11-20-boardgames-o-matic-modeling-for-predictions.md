---
layout: post
published: true
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

# The Approach
In general, here's what we will be performing in order with each modeling technique:

1. Acquire train and test sets
2. Normalize train set ratings
3. Fit train set on model instance
4. Make predictions on test set
5. Determine RMSE
6. Provide recommendations for specified user

First we get our regular imports in 
