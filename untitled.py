import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from collections import defaultdict
from scipy.spatial.distance import cdist
import time
cid = '41ef3d43ab644b70b02c5cd59c863774'
secret = 'e6dc9a5208a04977991022b24a1bb6fe'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager
=
client_credentials_manager)

df = pd.read_csv('fourtet.csv')

def find_song(name, artist):
    """
    This function returns a dataframe with data for a song given the name and artist.
    The function uses Spotipy to fetch audio features and metadata for the specified song.

    """

    song_data = defaultdict()
    results = sp.search(q='track: {} artist: {}'.format(name,
                                                        artist), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]

    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['artist'] = [artist]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):
    """
    Gets the song data for a specific song. The song argument takes the form of a dictionary with
    key-value pairs for the name and release year of the song.
    """

    try:
        song_data = spotify_data[(spotify_data['track_name'] == song['name'])
                                 & (spotify_data['artist_name'] == song['artist'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['artist'])


def get_mean_vector(song_list, spotify_data):
    """
    Gets the mean vector for a list of songs.
    """

    song_vectors = []
    number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
                   'loudness', 'speechiness', 'tempo']
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    """
    Utility function for flattening a list of dictionaries.
    """

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=10):
    """
    Recommends songs based on a list of previous songs that a user has listened to.
    """

    metadata_cols = ['track_name', 'artist_name']
    number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
                   'loudness', 'speechiness', 'tempo']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    spotify_data = spotify_data.drop(['popularity', 'Unnamed: 0'], axis=1)
    X = spotify_data.select_dtypes(np.number)
    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
    cluster_pipeline.fit(X)
    cluster_labels = cluster_pipeline.predict(X)
    spotify_data['cluster'] = cluster_labels
    scaler = cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['track_name'].isin(song_dict['name'])]
    df_recs = pd.DataFrame(rec_songs[metadata_cols])
    return df_recs

title = st.text_input('Song')
artist = st.text_input('Artist')
if title and artist:
    song_list = [{'name': title, 'artist': artist}]
    st.write(recommend_songs(song_list, df, 10))

button = st.button('Update features manually')

def user_input_features():
    danceability = st.sidebar.slider('Danceability', 0.000000, 0.980000, 0.000000, 0.01)
    energy = st.sidebar.slider('Energy', 0.000281, 0.999000, 0.000281, 0.01)
    acousticness = st.sidebar.slider('Acousticness', 0.000002, 0.996000, 0.000002, 0.01)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.000000, 0.984000, 0.000000, 0.01)
    liveness = st.sidebar.slider('Liveness', 0.020600, 0.993000, 0.020600, 0.01)
    loudness = st.sidebar.slider('Loudness', -37.114000, -1.987000, -37.114000, 0.1)
    speechiness = st.sidebar.slider('Speechiness', 0.000000, 0.947000, 0.000000, 0.01)
    tempo = st.sidebar.slider('Tempo', 0.000000, 210.029000, 0.000000, 1.0)
    valence = st.sidebar.slider('Valence', 0.000000, 0.996000, 0.000000, 0.01)

    user_data = {'danceability': danceability,
                'energy': energy,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'loudness': loudness,
                'speechiness': speechiness,
                'tempo': tempo,
                'valence': valence}

    features = pd.DataFrame(user_data, index=[0])
    return features

df_user = user_input_features()
if button:
    df_user

    df3 = pd.DataFrame()
    for k, v in df_user.iterrows():
        i = ((df['danceability']-v['danceability']) * \
           (df['loudness']-v['loudness']) * \
           (df['valence']-v['valence'])).abs().idxmin()
        df3 = df3.append(df.loc[i])
        df3 = df3.drop(['Unnamed: 0', 'popularity'], axis=1)
    df3

    new_song_list = [{'name': df3.iloc[0]['track_name'], 'artist': df3.iloc[0]['artist_name']}]
    st.write(recommend_songs(new_song_list, df, 10))