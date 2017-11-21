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
In [part 1](https://timmoti.github.io/2017-10-11-scraping-for-geek-data/), we scraped for data off [Boardgamegeek](http://boardgamegeek.com)(BGG). [Part 2](https://timmoti.github.io/2017-11-05-boardgames-o-matic-modeling-for-predictions/) saw us making predictions off the ratings matrix with 5 models and evaluating offline through RMSE and top 20 lists recommended for me.

In this final installment, we will be building a web app using [flask](http://flask.pocoo.org), a microframework based on python and deploying it via an Amazon EC2 instance. The aim is to get online evaluations from users of BGG.

We will be deploying recommendations from 3 algorithms, namely SVD with 50 latent factors, Non-negative matrix factorizaiton (NNMF) with 10 latent factors and the Cosine Similarity neighbourhood method.

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
The function is called into action whenever the `<a>` tag that directs to the '/about' page is triggered. It will then render the appropriate html page under its `render_template` function.

![header_html](/img/header_html.png)
*Example of header tag of each html page, where the `<a>` tag links to the Home, About and Contact pages*

The model section of the controller.py file houses the pickled and csv files that will be used to pass in as input into our recommend functions. These include the main ratings matrix, along with pre-calculated predictions of the SVD50 and NNMF models as well as the item-item similarity matrix from [part 2](https://timmoti.github.io/2017-11-05-boardgames-o-matic-modeling-for-predictions/).

The main bulk of the app's intelligence comes from the recommend function where it gets called several times depending on where the user has progressed on the app. The recommend function in the flask app is essentially a combination of the 3 recommend functions derived from the individual classes in my modeling work, tweaked to activate when certain conditions are met in the course of the flow of the app. You may delve deeper into the code at my [Github repo](https://github.com/timmoti/boardgames-o-matic).

At this point, it seems appropriate for me to talk about the flow of the app and what a user is prompted to do.

# App flow

![homepage](/img/homepage(161117).png)
*Homepage*

This is where a user would start. First, in order to get results, they need to have an account on the Boardgamegeek website, and have rated at least 10 games as of 13th October 2017, the date where I had finished scraping the data. After entering a valid username, they will be brought to the first list which was generated for them via the SVD model with 50 latent factors.

![list_top](/img/list_top.png)
*Top half of top 20 list*

Once they scroll down, they will be prompted to rate the list.

![list_bottom](/img/list_bottom.png)
*Yay or nay*

After rating, the user will be brought to a screen that will explain what method their list was generated from and will be prompted to click on the button below to generate a new list. The order of model generation is first SVD with 50 factors, then NNMF with 10 factors and finally Cosine Similarity.

![explanation](/img/explanation.png)
*Explains the method used and prompts the user to dive in for more*

Once a user has gone through all 3 lists, they will be brought to a page where they are able to revisit the lists again. The only difference is that they will not be able to re-rate them.

![rated](/img/rated.png)
*Check them out again.*

The ratings are stored in a txt file whereby the username and their rating for the model are comma-separated. The ratings are encoded to be 1 for liked the list and 0 for did not like the list.

One major issue I'd encountered while developing the app was in how variables are passed from the html page to the controller file and back again to another html page. I had to make use of hidden input tags on several html pages just to contain these variables for use further down the line.

Also, 

# Deploying the app

As we have some amazon credits given to us, I decided to put my web app up on an AWS EC2 instance. Unfortunately, as my 3 pickled files were at least 1.6GB in size, I've had to opt for a large vCPU instance with 16GB RAM and 4 vCPU cores.

The app's files are stored in a github repository where they are synced automatically between the AWS servers and Github's via a bash script utilizing crontabs.

# Marketing and getting feedback

I wanted to test the app out in the real world and posted a message on the BGG website itself under the Recommendations forum. The response was beyond what I had expected.

![forum](/img/forum.png)
*Simple and honest message*

I received over 240 unique users testing the app in 2 days

