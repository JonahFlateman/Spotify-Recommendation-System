#!/usr/bin/env python
# coding: utf-8

# In[24]:


import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
local_css('cssclient.css')

types_of_features = ("acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence")

st.title("Spotify Features app")
name_of_artist = st.text_input("Artist Name")
name_of_feat = st.selectbox("Feature", types_of_features)
button_clicked = st.button("OK")

from client import spotipyclient
import pandas as pd

client_id = 'YOUR CID'
client_secret = 'YOUR SECRET'

spotify = spotipyclient.SpotifyAPI(client_id, client_secret)

data = spotify.search({"artist": f"{name_of_artist}"}, search_type="track")

need = []
for i, item in enumerate(data['tracks']['items']):
    track = item['album']
    track_id = item['id']
    song_name = item['name']
    popularity = item['popularity']
    need.append((i, track['artists'][0]['name'], track['name'], track_id, song_name, track['release_date'], popularity))

track_df = pd.DataFrame(need, index=None, columns=('Item', 'Artist', 'Album Name', 'Id', 'Song Name', 'Release Date', 'Popularity'))

access_token = spotify.access_token

headers = {
    "Authorization": f"Bearer {access_token}"
}

endpoint = "https://api.spotify.com/v1/audio-features/"

import requests

feat_df = pd.DataFrame()
for id in track_df['Id'].iteritems():
    track_id = id[1]
    lookup_url = f"{endpoint}{track_id}"
    ra = requests.get(lookup_url, headers=headers)
    audio_feat = ra.json()
    features_df = pd.DataFrame(audio_feat, index=[0])
    feat_df = feat_df.append(features_df)
    
full_data = track_df.merge(feat_df, left_on="Id", right_on="id")

sort_df = full_data.sort_values(by=["Popularity"], ascending=False)

chart_df = sort_df[['Artist', 'Album Name', 'Song Name', 'Release Date', 'Popularity', f'{name_of_feat}']]

import altair as alt

feat_header = name_of_feat.capitalize()
st.header(f'{feat_header}' " vs. Popularity")
c = alt.Chart(chart_df).mark_circle().encode(
    alt.X('Popularity', scale=alt.Scale(zero=False)), y=f'{name_of_feat}', color=alt.Color('Popularity', scale=alt.Scale(zero=False)), 
    size=alt.value(200), tooltip=['Popularity', f'{name_of_feat}', 'Song Name', 'Album Name'])

st.altair_chart(c, use_container_width=True)

st.header("Table of Attributes")
st.table(chart_df)

st.write("acousticness: Confidence measure from 0.0 to 1.0 on if a track is acoustic.")
st.write("danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.")
st.write("energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.")
st.write("instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.")
st.write("liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.")
st.write("loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.")
st.write("speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.")
st.write("tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.")
st.write("valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).")


st.write("Information about features is from:  https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/")

