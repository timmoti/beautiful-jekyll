---
layout: post
published: false
title: 'Boardgames-O-Matic: Flask app and online evaluations'
subtitle: >-
  Part 3 of 3 where I build a board games recommender system for
  Boardgamegeek.com users
date: '2017-11-11'
image: /img/bgg-logo.jpg
bigimg: /img/title_darkened.png
---
In [part 1](https://timmoti.github.io/2017-10-11-scraping-for-geek-data/), we scraped for data off [Boardgamegeek](http://boardgamegeek.com). [Part 2](https://timmoti.github.io/2017-11-05-boardgames-o-matic-modeling-for-predictions/) saw us making predictions off the ratings matrix with 5 models and evaluating offline through RMSE and top 20 lists recommended for me.

In this final installment, we will be building a web app using [flask](http://flask.pocoo.org), a 
