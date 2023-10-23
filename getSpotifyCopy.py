
# Import libraries

import time
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials

# Connect to Spotify

# Insert your own
# cid = ''
# secret = ''

client_credentials_manager = SpotifyClientCredentials(client_id = cid, client_secret = secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Get tracks and track features

def getTrackID(user, playlist_id):
    id = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        id.append(track['id'])
    return id

def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)

    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']

    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    valence = features[0]['valence']
    tempo = features[0]['tempo']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, release_date, length, popularity, acousticness, danceability, 
             energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, time_signature]
    return track

def processTracks(userID, playlistID):
    ids = getTrackID(userID, playlistID)
    tracks = []
    
    for i in range(len(ids)):
        time.sleep(0.5)
        track = getTrackFeatures(ids[i])
        tracks.append(track)

    df = pd.DataFrame(tracks, columns=['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'acousticness', 'danceability', 
                                       'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'time_signature'])
    
    return df

# Save playlists to csv in order to easily get instead of using API

# Hockey  - 187
nhl17 = processTracks('63fa3b6e0c5d442f', '0fqsHCeHznQVNQKTp1n18G') #12
nhl17.to_csv('nhl17.csv', sep = ',')
time.sleep(0.5)
nhl19 = processTracks('8f793337760f4422', '5wgSim1qpFw7yuxsEdPKEy') #19
nhl19.to_csv('nhl19.csv', sep = ',')
time.sleep(0.5)
nhl20 = processTracks('632ffb9ac8df4f42', '3aVlPwvEeRXbbDKHRABniH') #18
nhl20.to_csv('nhl20.csv', sep = ',')
time.sleep(0.5)
nhl21 = processTracks('9d9496b4406a4821', '6wrh3jdJs4qOcECJORCIxf') #23
nhl21.to_csv('nhl21.csv', sep = ',')
time.sleep(0.5)
nhl22 = processTracks('814646d452534aed', '2fY8NDvI9UW3UzDkujn4yi') #42
nhl22.to_csv('nhl22.csv', sep = ',')
time.sleep(0.5)
nhl23 = processTracks('4c0efd19528b43bd', '41tgVNHUqHM11bbi6qvKmc') #40
nhl23.to_csv('nhl23.csv', sep = ',') 
time.sleep(0.5)
nhl24 = processTracks('9dae56cdfaaf4c91', '37i9dQZF1DWWP0WY5aV2Xz') #33
nhl24.to_csv('nhl24.csv', sep = ',')


# NFL - 199
nfl19 = processTracks('3f203b9ea37542ef', '2Q41eHDoyL0Aq7bQjIdEwl') #27
nfl19.to_csv('nfl19.csv', sep = ',')
time.sleep(0.5)
nfl20 = processTracks('ea167f84e2ef41c2', '5KSEn5wXudMFCiPzkEksLR') #18
nfl20.to_csv('nfl20.csv', sep = ',')
time.sleep(0.5)
nfl21 = processTracks('f0dee77e6094496f', '3bmmLo4IVjtYusLqqBMDe9') #20
nfl21.to_csv('nfl21.csv', sep = ',')
time.sleep(0.5)
nfl22 = processTracks('3412e752f776495d', '5PscZE6kBzgzfWRfybGRzR') #57
nfl22.to_csv('nfl22.csv', sep = ',')
time.sleep(0.5)
nfl23 = processTracks('07f43ef5c38546ce', '2sNmHV2Nn5yOCinuD92WHE')#38
nfl23.to_csv('nfl23.csv', sep = ',') 
time.sleep(0.5)
nfl24 = processTracks('3dd36611367b4c32', '3MO8i48TWpdhmjJf8m225L') #39
nfl24.to_csv('nfl24.csv', sep = ',')


# FIFA - 201
fifa21 = processTracks('7acf68f5408d4c06', '3EEtqDscUeqwVVdx6JuMKU') #43
fifa21.to_csv('fifa21.csv', sep = ',') 
time.sleep(0.5)
fifa22 = processTracks('2bc86a95317d43d5', '2a6OSlFchVcDonrXMMJ6EM') #58
fifa22.to_csv('fifa22.csv', sep = ',')
time.sleep(0.5)
fifa23 = processTracks('10e50533e95c4698', '37i9dQZF1DX4vgOVqe6BJn') #100
fifa23.to_csv('fifa23.csv', sep = ',')


# NBA - 194
nba2k22 = processTracks('1c474a1f645d4ebb', '7xnijvEtf5TkP0hzT0Md4B') #100
nba2k22.to_csv('nba2k22.csv', sep = ',')
time.sleep(0.5)
nba2k20 = processTracks('1c474a1f645d4ebb', '4wZ1wjXmLXHegWrLpkbPW4') #95
nba2k20.to_csv('nba2k20.csv', sep = ',')

# Total: 781







