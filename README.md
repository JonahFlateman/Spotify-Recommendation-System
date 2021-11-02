# Spotify Recommendation System and Streamlit Web App

This project entails building a recommendation system in Spotify which is deployed as a web app using Streamlit. We will outline the process for accessing the Spotify Web API to extract a playlist and build a model to provide top recommendations to a user based on similarity towards other songs and audio features. Using Streamlit, we will show how this model can be deployed and accessed with music recommendations embedded for a user to listen to within a web browser. We will be using a playlist from Four Tet, an electronic music producer and DJ.

## Goals

* Access the Spotify API, extract playlist metadata and audio features
* Identify features to be used in modeling and visualize each song within clustered groups
* Create binary classification model and visualize audio feature importances
* Create user interface to return song data from either Spotify API or dataset
* Build content-based recommendation system to return playlist songs based on user input
* Deploy system in Streamlit, give user ability to input song or adjust audio features and return Spotify songs embedded in browser

## Methodologies

* Use Spotipy to access Spotify API for converting playlist to DataFrame using Pandas and export as .csv file
* Scale numeric variables and cluster using KMeans, use Principal Component Analysis and fit data for visualized clusters using Plotly
* Plot confusion matrix using Gradient Boosting and SMOTE, visualize feature importances using LIME
* Use Spotipy search to extract individual songs/audio features
* Calculate cosine distance between mean vectors of extracted songs, return in song/artist DataFrame format
* Embed Four Tet playlist in Streamlit, use sidebar to display audio features for adjusting and return closest match in dataset for recommendation
* Add text boxes in Streamlit for up to three songs for user input, append track ID of results to Spotify URL for embedding

## Navigating This Repository

* [Using Spotify API](https://github.com/JonahFlateman/capstone/blob/main/fourtetplaylist.ipynb) - using Spotipy to retrieve playlist tracks by creator and playlist URI, obtaining track ID and audio features and converting to DataFrame
* [Data Wrangling, Scaling, Modeling and Clustering](https://github.com/JonahFlateman/capstone/blob/main/fourtetclustering.ipynb) - isolating and scaling numeric features using KMeans to cluster and Plotly to visualize, creating confusion matrix and LIME visualization of gradient boosted binary classification model
* [Building Recommendation System](https://github.com/JonahFlateman/capstone/blob/main/fourtetrecommender.ipynb) - using model and Spotipy, retrieve tracks and calculate vectors and cosine distances to build recommender
* [Streamlit App](https://github.com/JonahFlateman/capstone/blob/main/untitled.py) - code used for building web app, includes features for user input in text boxes and sidebar, description for audio features

## Analysis

### Obtaining Playlist Data
[Spotipy](https://spotipy.readthedocs.io/en/2.19.0/) is a Python library which allows developers access to the Spotify Web API upon input of their Client ID and Client Secret (these can be obtained through [Spotify For Developers](https://developer.spotify.com/)). For the purposes of this project and to build a more niche recommendation system, the playlist we will be downloading contains 1,723 songs - however this process can be emulated with any Spotify playlist using the creator's username and playlist URI (obtainable in Spotify).

### Clustering and Modeling

<img src="/images/threeclusters.png" width="400" height="300"/> <img src="/images/twoclusters.png" width="400" height="300"/>

Our final model used KMeans with three clusters, however the data points of a similar two-cluster model are an intriguing starting place for a classification model with an emphasis on feature importances. Our final model using SMOTE and GradientBoostingClassifier gives us 97.2% accuracy on the test set and identifes its feature importances as:

<img src="/images/featureimportances.png" width="400" height="300"/> <img src="/images/confusionmatrix.png" width="400" height="300"/>
<img src="/images/limevisual.png" width="800" height="300"/>

Energy, Danceability, and Acousticness are the highest; for our recommendation system and especially in the Streamlit app we will want a way to highlight how these features affect our results output to the user.

### Building and Deploying a Recommendation System

Our recommendation system obtains data for one or multiple songs with either Spotipy or from a dataset provided by the user. Separating out columns for our audio features, we calculate the mean vector of our song list as our "song center" and use cosine distance to find best matches to return as a DataFrame. The end result is a user-input song or song list returning n number of recommendations.

In Streamlit the model can give recommendations in two ways. First a user has the option to adjust audio features manually using a sidebar, the app will take the nearest match for its input and provide the recommendation. A user may also enter up to three songs and the code will execute. In both cases the Spotify-embedded tracks will return and the user can listen in-app.

[Access the Streamlit app here](https://share.streamlit.io/jonahflateman/spotify-recommendation-system/main)




