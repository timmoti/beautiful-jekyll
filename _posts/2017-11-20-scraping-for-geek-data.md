---
layout: post
published: true
title: Scraping for geek data
subtitle: Part 1 of 3 where I build a board games recommender system for BGG.com users
date: '2017-10-11'
image: /img/bgg-logo.jpg
bigimg: /img/Game-Geek.png
---
[Boardgamegeek](https://boardgamegeek.com/) is board game nirvana for geeks, nerds and the occasional cool person. It is like Wikipedia to your science. This website holds a massive library of more than 90,000 games, with about 800,000 registered users and more than 150,000 active monthly users. 

What is more impressive about this site is how open the data is. You can view anyone's boardgame collection, how they have rated their games, how often they have been playing and even their wishlist of games to buy and/or try. The notion that this website is like the swiss army knife of the boardgaming world makes it easy to entrust it with everything boardgame-related in one's life.

# Yummy, yummy ratings...

With the wealth of games and the dearth of time, a good boardgame recommender seems like the ideal solution to helping one choose wisely. The beauty of Boardgamegeek (or BGG for short) is its rating system. Users can rate games on a scale of 1 to 10. There is even a rubric to suggest how you might want to rate your games. It is entirely up to you however and some users even give ratings to 2 decimal places!

![Bgg ratings](/img/BGG_ratings.png)
*Rate your games accordingly*

What's more, the ratings are colour coded on the webpage for quick identification. 

>Bloody-red belongs to a rating of 1 and it gets lighter till rating 5 where it turns purple. 
The transition from a sky-blue 7-rating and a light-green 8-rating is distinct, only to get to its darkest shade of green at 10.

# Scraping and calling

There exists a python module that allows one access to BGG's XML APIs via pythonic methods and classes. It unfortunately is not versatile enough for my needs. I will be scraping the page of the website where I sort all games in descending order of number of ratings. This means the game where most users have rated it will be right at the top. I will then scrape this page to gather the list of games up till the point where I hit the game with just less than 1000 ratings. This will be my game list dataset.

I will then be making API calls to gather the rating data for each game in the game list, as well as the users that have provided them. This will become my ratings list. I will then process this ratings list into an N x M matrix of N rows of users by M columns of games. The values in the matrix will be the ratings provided by users for games.

# Regular imports

We will import the usual python modules for data science work along with Beautiful Soup for HTML extraction and sleep to make sure we don't be a nuisance to the BGG servers.

```python
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from time import sleep


%config InlineBackend.figure_format = 'retina'
%matplotlib inline

sns.set_style('white')
```