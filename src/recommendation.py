import pandas as pd
import math
import ast
import csv


def calculate_edge_weight(track_u, track_v, weights):
    """
    Calculate the weight between two tracks based on multiple criteria.

    Parameters:
    - track_u, track_v: Dictionaries representing track details.
    - weights: Dictionary containing weights for different criteria.

    Returns:
    - weight: The computed similarity weight.
    """
    # Artist similarity
    artist_sim = 0.5 if track_u['artist_id'] == track_v['artist_id'] else 0
    artist_weight = weights['artist']

    # Genre similarity
    try:
        genre_u = set(ast.literal_eval(track_u['genre'])) if pd.notna(track_u['genre']) else set()
        genre_v = set(ast.literal_eval(track_v['genre'])) if pd.notna(track_v['genre']) else set()
        genre_intersection = genre_u.intersection(genre_v)
        # Penalize a bit large sets
        genre_sim = len(genre_intersection) if len(genre_u) == len(genre_v) \
                    else len(genre_intersection) - 0.5*(max(len(genre_u), len(genre_v)) - len(genre_intersection))
        feature_weight = weights['feature']

    except Exception as e:
        print(f"Error parsing genre for tracks {track_u['track_id']} and {track_v['track_id']}: {e}")
        genre_sim = 0
    genre_weight = weights['genre']

    # Temporal proximity (based on release date difference)
    time_diff = abs(track_u['release_date'] - track_v['release_date'])
    time_sim = 1 / (1 + math.exp(0.1 * time_diff))  # Logistic decay function
    time_weight = weights['time'] * math.log(time_diff + 1) if time_diff else weights['time']


    # Topic similarity
    topic_u = {ast.literal_eval(track_u[f'topic_{i}'])[0] for i in range(1, 4)}
    topic_v = {ast.literal_eval(track_v[f'topic_{i}'])[0] for i in range(1, 4)}
    topic_intersection = len(topic_u.intersection(topic_v))
    topic_sim = topic_intersection
    topic_weight = weights['topic']

    # Feature similarity
    feature_u = {ast.literal_eval(track_u[f'feature_{i}'])[0] for i in range(1, 4)}
    feature_v = {ast.literal_eval(track_v[f'feature_{i}'])[0] for i in range(1, 4)}
    feature_intersection = len(feature_u.intersection(feature_v))
    feature_sim = feature_intersection


    if track_u['feature_1'][0] == track_v['feature_1'][0]:
        feature_sim = feature_sim + 0.5
        if track_u['feature_2'][0] == track_v['feature_2'][0]:
            feature_sim = feature_sim + 0.5
            if track_u['feature_3'][0] == track_v['feature_3'][0]:
                feature_sim = feature_sim + 1

    # Compute final weight as a weighted sum of all criteria
    weight = (
        artist_weight * artist_sim +
        genre_weight * genre_sim +
        time_weight * time_sim +
        topic_weight * topic_sim +
        feature_weight * feature_sim
    )

    return weight

# Function to get topics and features by track ID
def get_topics_and_features_by_track_id(track_id, tracks_data):
    """
    Retrieve topics and features for a given track ID.

    Parameters:
    - track_id: Track ID to search for.
    - tracks_data: DataFrame containing track details.

    Returns:
    - A dictionary with topics and features or None if track ID is not found.
    """
    filtered_data = tracks_data[tracks_data['track_id'] == track_id]
    if not filtered_data.empty:
        row = filtered_data.iloc[0]
        topics = [row[f'topic_{i}'] for i in range(1, 4)]
        features = [row[f'feature_{i}'] for i in range(1, 4)]
        return {'topics': topics, 'features': features}
    else:
        print(f"Track ID {track_id} not found.")
        return None


def recommendation_system(initial_query_id, tracks_data, weights, max_playlist_size=100, max_songs_per_artist=1):
    """
    Generate a playlist using a hybrid recommendation approach and save it to a dynamically named file.

    Parameters:
    - initial_query_id: Starting track ID.
    - tracks_data: DataFrame of track data.
    - weights: Weights for criteria.
    - max_playlist_size: Maximum playlist size.
    - max_songs_per_artist: Max songs per artist in the playlist.

    Returns:
    - Final playlist of recommended tracks.
    """
    exclude_ids = set()
    playlist = []
    artist_song_count = {}

    # Get recommendations for the current track
    query_track = tracks_data[tracks_data['track_id'] == initial_query_id].iloc[0].to_dict()
    recommendations_with_weights = []
    for _, candidate_track in tracks_data.iterrows():
        candidate_track = candidate_track.to_dict()
        if candidate_track['track_id'] in exclude_ids or candidate_track['track_id'] == initial_query_id:
            continue
        weight = calculate_edge_weight(query_track, candidate_track, weights)
        recommendations_with_weights.append((candidate_track['track_id'], weight))

    # Sort recommendations by descending weight
    recommendations_with_weights.sort(key=lambda x: x[1], reverse=True)

    # Add tracks to playlist
    while len(playlist) < max_playlist_size:
        if not recommendations_with_weights:
            print("No more recommendations available.")
            break
        top_recommendation = recommendations_with_weights.pop(0)
        track_id, weight = top_recommendation

        # Get artist_name and track_name
        track_details = tracks_data[tracks_data['track_id'] == track_id].iloc[0]
        artist_name = track_details['artist_name']
        if artist_name not in artist_song_count:
            artist_song_count[artist_name] = 0
        if artist_song_count[artist_name] >= max_songs_per_artist:
            continue

        # Add track to playlist
        playlist.append((track_id, weight))  # Store track ID with weight
        exclude_ids.add(track_id)
        artist_song_count[artist_name] += 1

    # Prepare the playlist output
    final_playlist = []

    # Add the initial query track to the first line
    query_topics = get_topics_and_features_by_track_id(initial_query_id, tracks_data)
    query_details = tracks_data[tracks_data['track_id'] == initial_query_id].iloc[0]
    initial_artist = query_details['artist_name']
    initial_song = query_details['track_name']
    initial_line = {
        'Score': None,  # No score for the initial track
        'Track ID': initial_query_id,
        'Artist': initial_artist,
        'Song': initial_song,
        'Genre': query_details['genre'],
        'Release Date': query_details['release_date'],
        'Topic_1': query_topics['topics'][0] if query_topics else None,
        'Topic_2': query_topics['topics'][1] if query_topics else None,
        'Topic_3': query_topics['topics'][2] if query_topics else None,
        'Feature_1': query_topics['features'][0] if query_topics else None,
        'Feature_2': query_topics['features'][1] if query_topics else None,
        'Feature_3': query_topics['features'][2] if query_topics else None,
    }
    final_playlist.append(initial_line)

    # Add the recommended tracks to the playlist
    for track_id, weight in playlist[:max_playlist_size]:
        track_details = tracks_data[tracks_data['track_id'] == track_id].iloc[0]
        if track_details['artist_name'] == initial_artist:
            continue
        artist_name = track_details['artist_name']
        track_name = track_details['track_name']
        genre = track_details['genre']
        release_date = track_details['release_date']
        topics_features = get_topics_and_features_by_track_id(track_id, tracks_data)
        playlist_line = {
            'Score': weight,  # Add the score (edge weight)
            'Track ID': track_id,
            'Artist': artist_name,
            'Song': track_name,
            'Genre': genre,
            'Release Date': release_date,
            'Topic_1': topics_features['topics'][0] if topics_features else None,
            'Topic_2': topics_features['topics'][1] if topics_features else None,
            'Topic_3': topics_features['topics'][2] if topics_features else None,
            'Feature_1': topics_features['features'][0] if topics_features else None,
            'Feature_2': topics_features['features'][1] if topics_features else None,
            'Feature_3': topics_features['features'][2] if topics_features else None,
        }
        final_playlist.append(playlist_line)

    # Create dynamic file name
    safe_artist_name = "".join(c if c.isalnum() or c in (" ", "-", "_")
                              else "" for c in initial_artist).replace(" ", "_")
    safe_track_name = "".join(c if c.isalnum() or c in (" ", "-", "_")
                              else "" for c in initial_song).replace(" ", "_")
    output_file = f'data/generated_playlist_from_{safe_artist_name}_{safe_track_name}.csv'

    # Save the playlist to a CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=final_playlist[0].keys())
        writer.writeheader()
        writer.writerows(final_playlist)

    print(f"Playlist saved to {output_file}")
    return final_playlist


# Define weights for each criterion
weights = {
    'artist': 3,
    'genre': 6,
    'time': 2,
    'topic': 3,
    'feature': 5
}

# Load vectorized track data
file_path = 'data/final_processed_music_data.csv'
tracks_data = pd.read_csv(file_path)

# Retain only the relevant columns
columns_to_use = [
    'artist_id', 'track_id', 'genre', 'release_date', 'track_name', 'artist_name',
    'topic_1', 'topic_2', 'topic_3',
    'feature_1', 'feature_2', 'feature_3'
]
tracks_data = tracks_data[columns_to_use]

# Demander à l'utilisateur de saisir un track_id valide entre 1 et 28372
try:
    initial_track_id = int(input("Entrez l'ID du morceau de départ (entre 1 et 28372) : "))
    if initial_track_id < 1 or initial_track_id > 28372:
        raise ValueError("L'ID doit être compris entre 1 et 28372.")
except ValueError as e:
    raise ValueError("Entrée invalide : {}".format(e))

# Appel du système de recommandation
recommended_playlist = recommendation_system(
    initial_query_id=initial_track_id,
    tracks_data=tracks_data,
    weights=weights,
    max_playlist_size=100
)
