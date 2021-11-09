# Spotify Recommendation System and Streamlit Web App

## Introduction

Spotify creators and artists rely on user interaction to increase traffic on their profiles. In adddition to song plays, artists often make playlists of personally-recommended songs to engage with their audience. For users, Spotify may recommend its own playlists, similar artists or songs based on other user metrics or artist/song similarities. This project seeks to increase direct user interaction with artists in Spotify by building a content-based recommendation system based on songs in a artist's playlist. The user will input any songs of their choosing or manually adjust Spotify audio features, and the recommender will return a list of similar songs from the playlist based on the numerical data of audio features as defined by Spotify. By suggesting playlist songs with the most similarities to a user-suggested track we create a better experience for the user and increase traffic to the artist's playlist and Spotify profile.

## Data

[Raw Data](https://github.com/JonahFlateman/capstone/blob/main/fourtet.csv)

We will be using a 1700+ row dataset collected from the Spotify API and consisting of songs from a playlist from DJ and producer Four Tet, which can be found in the above CSV file. The dataset consists of descriptive metadata (track name, artist name) as well as numerical audio features which we will use for modeling (acousticness, danceability, duration, energy, instrumentalness, liveness, loudness, tempo and valence).

We will test our models on two additional smaller playlists to evaluate its feasability with different-sized datasets - Spotify's ["Songs to Sing in the Shower"](https://github.com/JonahFlateman/capstone/blob/main/showersongs.csv) (200 rows) and Max Richter's ["Kitchen Playlist"](https://github.com/JonahFlateman/capstone/blob/main/maxrichter.csv) (78 rows).


## Navigating This Repository

* [Using Spotify API](https://github.com/JonahFlateman/capstone/blob/main/playlist_to_dataframe.ipynb) - using Spotipy to retrieve playlist tracks by creator and playlist URI, obtaining track ID and audio features and converting to DataFrame
* [Data Wrangling, Scaling, Modeling and Clustering](https://github.com/JonahFlateman/capstone/blob/main/clustering_and_modeling.ipynb) - isolating and scaling numeric features using KMeans to cluster and Plotly to visualize, creating confusion matrix and LIME visualization of Logistic Regression binary classification model
* [Building Recommendation System](https://github.com/JonahFlateman/capstone/blob/main/recommender.ipynb) - using model and Spotipy, retrieve tracks and calculate vectors and cosine distances to build recommender
* [Streamlit App](https://github.com/JonahFlateman/capstone/blob/main/streamlit_app.py) - code used for building web app, includes features for user input in text boxes and sidebar, description for audio features

## Analysis

### Obtaining Playlist Data

We first access the Spotify API in order to extract playlist metadata. [Spotipy](https://spotipy.readthedocs.io/en/2.19.0/) is a Python library which allows developers access to the Spotify Web API upon input of their Client ID and Client Secret (these can be obtained through [Spotify For Developers](https://developer.spotify.com/)). We convert our data to a Pandas DataFrame and save as a .csv file in the repository; this process can be emulated with any Spotify playlist using the creator's username and playlist URI (obtainable in Spotify). We now have a dataset ready for preprocessing and modeling.

### Clustering and Modeling

We select a subset of our data containing the numerical audio features mentioned above. To accurately discern if a content-based recommendation system is viable for this dataset, we can cluster and visualize our data points in a two-dimensional space using KMeans to determine the optimal number of clusters. We scale our data and use dimensionality reduction to visualize the data points in a two-dimensional space. The below graphs show each song represented as a data point in three-cluster and two-cluster scatter plots.

<img src="/images/threeclusters.png" width="400" height="300"/> <img src="/images/twoclusters.png" width="400" height="300"/>

For this particular dataset, the three-cluster plot is optimal for our content-based recommendation system; however we can use the two-cluster plot to create a classifier to predict which song ends up in which cluster and the feature importances of this classification. Knowing this will help us determine the effects of this particular playlist on our recommendation system and how it might function with other playlist of different character and type.

<img src="/images/featureimportances.png" width="400" height="300"/> <img src="/images/confusionmatrix.png" width="400" height="300"/>
<img src="/images/limevisual.png" width="800" height="300"/>

A Logistic Regression classification model yields 98.2% accuracy on an untrained set of data. Visualizing our feature importances shows acousticness and energy as most important, and we can use a LIME visual on any track in the playlist to determine the importance of each feature on its classification. 

### Building a Recommendation System

Based on the visualized data points of the above three-cluster model, a content-based recommendation system will work for this playlist. Our system allows a user to input one or multiple songs of their choosing in Spotify; they will then be recommended similar songs from the artist's playlist. The user can select any songs as long they exist in Spotify's database for their recommendations.

Separating out audio features in our dataset, we calculate the mean vector of a song list as our "song center" and use cosine distance to find closest matches to return as a DataFrame. The end result is a user-input song or song list returning closest-distance song suggestions for the user based on Spotify's numerical audio features.

### Deployment in Streamlit

We deploy the model using a Streamlit web app. In Streamlit the model can give recommendations in two ways. First a user has the option to adjust audio features manually using a sidebar; the app will search the existing playlist, find the song with the closest audio features for its input and provide the recommendation. A user may also enter up to three songs of their choosing, and the app will search Spotify for its data. In both cases the Spotify-embedded tracks will return and the user can listen in-app.

[Access the Streamlit app here](https://share.streamlit.io/jonahflateman/spotify-recommendation-system/main)

### Business Recommendations

With accurate song recommendations from an artist's playlist, we increase traffic for the artist and better engage Spotify users in song selection. While the current model is deployed in a Streamlit app, it could be applied within Spotify's desktop and mobile apps for users to get direct song recommendations and connect with their favorite artists. While Spotify currently suggests similar artists or auto-plays tracks for users based on the current artists, a hybrid recommendation system should be able to allow for the user to discover new music while highlighting the artist's presence in these recommendations.

### Conclusions and Future Work

Our model provides accurate playlist song recommendations with the 1700+ row dataset given input of any song in Spotify. Evaluating it on smaller playlists (200 and 78 rows respectively) yielded accuracies of 84% and 90% on untrained datasets. For future recommendation systems we will need to adjust how the model is grouped in clusters and tune it to accomodate playlists of different sizes. We used audio feature metrics to build our recommendation system, however Spotify does provide additional metrics such as popularity which could skew the data on future models. For larger playlists, idenfitying our most important features and removing certain features might be needed to increase accuracy, especially on more genre-specific playlists with closer song similarities.
