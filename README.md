# Spotify Recommendation System and Streamlit Web App

This project entails building a recommendation system in Spotify which is deployed as a web app using Streamlit. We will outline the process for accessing the Spotify Web API to extract a playlist and build a model to provide top recommendations to a user based on similarity towards other songs and audio features. Using Streamlit, we will show how this model can be deployed and accessed with music recommendations embedded for a user to listen to within a web browser. We will be using a playlist from Four Tet, an electronic music producer and DJ.

## Goals

* Access the Spotify API, extract playlist metadata and audio features
* Identify features to be used in modeling and visualize each song within clustered groups
* Create user interface to return song data from either Spotify API or dataset
* Build content-based recommendation system to return playlist songs based on user input
* Deploy system in Streamlit, give user ability to input song or adjust audio features and return Spotify songs embedded in browser

## Methodologies

* Use Spotipy to access Spotify API for converting playlist to DataFrame using Pandas and export as .csv file
* Scale numeric variables and cluster using KMeans, use Principal Component Analysis and fit data for visualized clusters using Plotly
* Use Spotipy search to extract individual songs/audio features
* Calculate cosine distance between mean vectors of extracted songs, return in song/artist DataFrame format
* Embed Four Tet playlist in Streamlit, use sidebar to display audio features for adjusting and return closest match in dataset for recommendation
* Add text boxes in Streamlit for up to three songs for user input, append track ID of results to Spotify URL for embedding

## Navigating This Repository

* [Retrieving Playlist](https://github.com/JonahFlateman/capstone/blob/main/fourtetplaylist.ipynb)
* Using Spotipy to retrieve playlist tracks by creator and playlist URI, obtaining track ID and audio features and converting to DataFrame
