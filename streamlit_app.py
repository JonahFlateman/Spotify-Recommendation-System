import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
#from collections import defaultdict
from scipy.spatial.distance import cdist
import time

cid = st.secrets['client_id']
secret = st.secrets['client_secret']
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager
=
client_credentials_manager)

df = pd.read_csv('fourtet.csv')

def find_song(name, artist):
    # Initialize an empty dictionary to store features and values
    song_data = defaultdict()

    # Using Spotipy search function for track and artist, returning None if cannot be found in Spotify
    results = sp.search(q='track: {} artist: {}'.format(name,
                                                        artist), limit=1)
    if results['tracks']['items'] == []:
        return None

    # Isolating track information and ID from results
    results = results['tracks']['items'][0]
    track_id = results['id']

    # Obtaining audio features
    audio_features = sp.audio_features(track_id)[0]

    # Preparing columns and converting to DataFrame
    song_data['name'] = [name]
    song_data['artist'] = [artist]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):
    # Function will attempt to find ID track name and artist from dataset to return track data
    try:
        song_data = spotify_data[(spotify_data['track_name'] == song['name'])
                                 & (spotify_data['artist_name'] == song['artist'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['artist'])


def get_mean_vector(song_list, spotify_data):
    # Initialize empty list to store vectors
    song_vectors = []

    # Identify audio features columns
    number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
                   'loudness', 'speechiness', 'tempo']

    # Append list of values to list
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} not found in Spotify or database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

        # Convert to single array and return mean
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def recommend_songs(song_list, spotify_data, n_songs=10):
    # Establishing metadata and numerical columns
    metadata_cols = ['track_name', 'artist_name']
    number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
                   'loudness', 'speechiness', 'tempo']

    # Getting mean vector
    song_center = get_mean_vector(song_list, spotify_data)

    # Dropping extra columns
    spotify_data = spotify_data.drop(['popularity', 'Unnamed: 0'], axis=1)

    # Using KMeans to cluster data, fitting and adding labels to dataset
    X = spotify_data.select_dtypes(np.number)
    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=3))])
    cluster_pipeline.fit(X)
    cluster_labels = cluster_pipeline.predict(X)
    spotify_data['cluster'] = cluster_labels

    # Scaling and transforming numerical columns of data and reshaped song center
    scaler = cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    # Computing cosine distance on transformed arrays
    distances = cdist(scaled_song_center, scaled_data, 'cosine')

    # Return sorted list of top n indices
    index = list(np.argsort(distances)[:, :n_songs][0])

    # Converting to DataFrame and returning track and artist name
    rec_songs = spotify_data.iloc[index]
    df_recs = pd.DataFrame(rec_songs[metadata_cols])
    return df_recs

st.title('Recommendations from Four Tet')
st.write('Generate song recommendations from DJ and producer Four Tet, '
         'based on his popular Spotify playlist.')

components.iframe("https://open.spotify.com/embed/playlist/2uzbATYxs9V8YQi5lf89WG", width=700, height=300)

st.write('## How It Works')
st.write('Fill in up to three songs and artists of your choice, or use the sidebar to adjust audio features '
         'on your own. You will be able to listen in your browser to the recommended songs! ')
st.write('Tip: Try entering hip-hop, dance, R&B or jazz tracks - the playlist has plenty of them.')

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
button1 = st.sidebar.button('Recommend Songs')
if button1:
    df3 = pd.DataFrame()
    for k, v in df_user.iterrows():
        i = ((df['danceability']-v['danceability']) * \
            (df['energy']-v['energy']) * \
            (df['acousticness']-v['acousticness']) * \
            (df['instrumentalness'] - v['instrumentalness']) * \
            (df['liveness'] - v['liveness']) * \
            (df['loudness'] - v['loudness']) * \
            (df['speechiness'] - v['speechiness']) * \
            (df['tempo'] - v['tempo']) * \
            (df['valence'] - v['valence'])).abs().idxmin()
        df3 = df3.append(df.loc[i])
        df3 = df3.drop(['Unnamed: 0', 'popularity'], axis=1)
    new_song_list = [{'name': df3.iloc[0]['track_name'], 'artist': df3.iloc[0]['artist_name']}]
    new_song_recs = recommend_songs(new_song_list, df, 5)
    for i, j in new_song_recs.itertuples(index=False):
        embed_string = 'https://open.spotify.com/embed/track/'
        id_list = []
        try:
            results = sp.search(q='track: {} artist: {}'.format(i, j), limit=1)
            id_list.append(results['tracks']['items'][0]['id'])
        except IndexError:
            continue
        concat_list = [embed_string + k for k in id_list]
        try:
            components.iframe(concat_list[0], width=700, height=300)
        except IndexError:
            continue

song_list = []
title = st.text_input('Song #1')
artist = st.text_input('Artist #1')
song_list.append({'name': title, 'artist': artist})
title2 = st.text_input('Song #2')
artist2 = st.text_input('Artist #2')
song_list.append({'name': title2, 'artist': artist2})
title3 = st.text_input('Song #3')
artist3 = st.text_input('Artist #3')
song_list.append({'name': title3, 'artist': artist3})
button2 = st.button('Go')
if button2:
    try:
        song_recs = recommend_songs(song_list, df, 5)
    except ValueError:
        st.markdown('**Song not found in Spotify, please try again**')
    for i, j in song_recs.itertuples(index=False):
        embed_string = 'https://open.spotify.com/embed/track/'
        id_list = []
        try:
            results = sp.search(q='track: {} artist: {}'.format(i, j), limit=1)
            id_list.append(results['tracks']['items'][0]['id'])
        except IndexError:
            continue
        concat_list = [embed_string + k for k in id_list]
        try:
            components.iframe(concat_list[0], width=700, height=300)
        except IndexError:
            continue



st.markdown('**Feature Descriptions**')
st.markdown('**Danceability** describes how suitable a track is for dancing based '
            'on a combination of musical elements including tempo, rhythm stability, beat strength, and '
            'overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.')
st.markdown('**Energy** is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and '
            'activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal '
            'has high energy, while a Bach prelude scores low on the scale. Perceptual features '
            'contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, '
            'and general entropy.')
st.markdown('**Acousticness** is a confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 '
            'represents high confidence the track is acoustic.')
st.markdown("**Instrumentalness** predicts whether a track contains no vocals. 'Ooh' and 'ahh' sounds are treated "
            "as instrumental in this context. Rap or spoken word tracks are clearly 'vocal'. The closer the "
            "instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. "
            "Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the "
            "value approaches 1.0.")
st.markdown('**Liveness** detects the presence of an audience in the recording. Higher liveness values represent '
            'an increased probability that the track was performed live. A value above 0.8 provides strong '
            'likelihood that the track is live.')
st.markdown('**Loudness** is the overall loudness of a track in decibels (dB). Loudness values are averaged '
            'across the entire track and are useful for comparing relative loudness of tracks. Loudness is '
            'the quality of a sound that is the primary psychological correlate of physical strength (amplitude). '
            'Values typical range between -60 and 0 db.')
st.markdown('**Speechiness** detects the presence of spoken words in a track. The more exclusively speech-like '
            'the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values '
            'above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and '
            '0.66 describe tracks that may contain both music and speech, either in sections or layered, including '
            'such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.')
st.markdown('**Tempo** is the overall estimated tempo of a track in beats per minute (BPM). In musical terminology, '
            'tempo is the speed or pace of a given piece and derives directly from the average beat duration.')
st.markdown('**Valence** is a measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. '
            'Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with '
            'low valence sound more negative (e.g. sad, depressed, angry).')
st.write('Tip: After setting your parameters, try adjusting just one or two to see if your results are different. '
         'Energy has a surprisingly large effect on the results!')
