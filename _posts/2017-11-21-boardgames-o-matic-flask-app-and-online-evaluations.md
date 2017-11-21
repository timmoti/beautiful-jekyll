---
layout: post
published: true
title: 'Boardgames-O-Matic: Flask app and online evaluations'
subtitle: >-
  Part 3 of 3 where I build a board games recommender system for
  Boardgamegeek.com users
date: '2017-11-11'
image: /img/bgg-logo.jpg
bigimg: /img/title_darkened.png
---
In [part 1](https://timmoti.github.io/2017-10-11-scraping-for-geek-data/), we scraped for data off [Boardgamegeek](http://boardgamegeek.com)(BGG). [Part 2](https://timmoti.github.io/2017-11-05-boardgames-o-matic-modeling-for-predictions/) saw us making predictions off the ratings matrix with 5 models and evaluating offline through RMSE and top 20 lists recommended for me.

In this final installment, we will be building a web app using [flask](http://flask.pocoo.org), a microframework based on python and deploying it via an Amazon EC2 instance. The aim is to get online evaluations from users of BGG.

We will be deploying recommendations from 3 algorithms, namely SVD with 50 latent factors, ALS with 10 latent factors and the Cosine Similarity neighbourhood method.

# Developing the flask app

![bgg_controllerpy](/img/bgg_controllerpy.png)
*A snapshot of the controller file*

Flask makes use of the MVC framework in web development to deploy small, lightweight web apps that are based on the Python language. Although it can also be used for larger scale settings, it is generally not considered a production-level tool. Nevertheless, it is well-suited for our purposes.

The controller.py file seen above controls the logic of how your html files are routed from one webpage to the next. The first thing to do is to `import flask`, followed by the following config command `app = flask.Flask(__name__)`. You can also choose to import certain modules from flask, which means you do not have to consistently type in the `flask` instance every time you perform a flask related function.

If you look at the diagram, the routes section of the controller file contains similar `@app.route` commands that control which html page gets rendered once it's called. Here's a sample of the route function that controls the logic for my about page.

```python
@app.route('/about')
def about():
	return flask.render_template('bgg_about.html')
```

