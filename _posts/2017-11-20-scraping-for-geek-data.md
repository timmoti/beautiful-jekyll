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

# Getting ready for the big scrape

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

In order to make sure that our http requests get sent even though we get disconnected from the network, I have here a function that helps to make our requests robust.

```python
def request(msg, slp=1):
    '''A wrapper to make robust https requests.'''
    status_code = 500  # Want to get a status-code of 200
    while status_code != 200:
        sleep(slp)  # Don't ping the server too often
        try:
            r = requests.get(msg)
            status_code = r.status_code
            if status_code != 200:
                print "Server Error! Response Code {}. Retrying...".format(r.status_code)
        except:
            print "An exception has occurred, probably a momentory loss of connection. Waiting one seconds..."
            sleep(1)
    return r
```

# The Scrape
Looking at the list of games sorted in descending order, I will extract games that have at least 1000 ratings. This is so that I would have sufficient information about each game to generate the recommendations.

![Games sorted in descending order of number of ratings](/img/sorted_games.png)
*Column highlighted in red indicates number of ratings a game has. Page taken from the BGG.com website*

## Making sense of the HTML code
In the following image, you'll see how I deconstructed the HTML in order to get what I want from each game, namely the rank, ID, name of game as well as the number of ratings.

Each game is first encapsulated around the <tr id='row_'> tags, so we make sure we look out for those.

Next, we extract the rank, ID and name of each game by identifying the next 3 <a> tags and perform some string processing techniques.
  
Finally, we acquire the number of ratings by looking at all <tr> tags with class attribute "collection_bggrating" and perform some more string processing techniques on them.

![scraping](/img/scraping.png)
*Output of scraping that makes sense by reading together with code below*

The commented code for getting the scraping results is right here:

```python
# Initialize a DF to hold all our scraped game info
games = pd.DataFrame(columns=["gameid", "gamename", "nratings","gamerank"])
min_nratings = 100000 # Set to a high number to satisfy while condition
npage = 1

# Scrape successful pages in the results until we get down to games with < 1000 ratings each
while min_nratings > 1000:
    # Get full HTML for a specific page in the full listing of boardgames sorted by 
    r = request("https://boardgamegeek.com/browse/boardgame/page/{}?sort=numvoters&sortdir=desc".format(npage))
    soup = BeautifulSoup(r.text, "html.parser")    
    
    # Get rows for the table listing all the games on this page. 100 per page
    table = soup.find_all("tr", attrs={"id": "row_"})
    # DF to hold this page's results
    temp_df = pd.DataFrame(columns=["gameid", "gamename", "nratings", "gamerank"], index=range(len(table)))  
    
    # Loop through each row and pull out the info for that game
    for idx, row in enumerate(table):
        links = row.find_all("a")
        try:
            gamerank = links[0]['name'] #Get rank of game
        except Exception: # Expansions will not have ranks and will be recorded as NaN rows in our dataframe
            continue
        gamelink = links[1]  # Get the relative URL for the specific game
        gameid = int(gamelink["href"].split("/")[2])  # Get the game ID by parsing the relative URL
        gamename = links[2].contents[0]  # Get the actual name of the game as the link contents

        ratings_str = row.find_all("td", attrs={"class": "collection_bggrating"})[2].contents[0]
        #split on white space to leave list of string of number, join on empty space then change to int datatype
        nratings = int("".join(ratings_str.split()))
        temp_df.iloc[idx, :] = [gameid, gamename, nratings, gamerank] #Add to temp_df
        
    # Concatenate the results of this page to the master dataframe
    min_nratings = temp_df["nratings"].min()  # The smallest number of ratings of any game on the page
    print "Page {} scraped, minimum number of ratings was {}".format(npage, min_nratings)
    games = pd.concat([games, temp_df], axis=0)
    npage += 1
    sleep(2) # Keep the BGG server happy.
```

The output helps keep our sanity in check and ensures we are on the right track.

![scraping output](/img/scraping_output.png)
*Notice we actually reached slightly below 1000 in our final total minimum ratings but that's ok. More data never did harm anyone.*

# Analyze our games dataset
From our output, we know that we have scraped 21 pages and if you were to refer to the image of the BGG website, the top right corner indicates that there are 941 pages of games. It might seem that we have only 