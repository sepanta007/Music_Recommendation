import pandas as pd

import math

# Function to calculate the similarity weight between two tracks
def calculate_edge_weight(track_u, track_v, weights):
    """
    Calculates the weighted similarity between two tracks with dynamic weight adjustments based on conditions.

    Parameters:
    - track_u, track_v: Dictionaries representing the tracks.
    - weights: Dictionary of weights for each criterion.

    Returns:
    - The weighted similarity value.
    """

    # Criterion: Artist similarity (using distance between artists, the farther apart, the lower the weight)
    artist_distance = abs(track_u['normalized_artist_id'] - track_v['normalized_artist_id'])
    artist_sim = 1 / (1 + artist_distance)  # Using inverse to penalize large distances

    # Adjust the artist weight based on distance (greater distance, less similarity)
    if artist_distance == 0:
        artist_weight = weights['artist'] * 2  # High similarity
    else:
        artist_weight = weights['artist']  # Less similarity

    # Criterion: Genre similarity with non-linear weight adjustment
    genre_sim = 1 if track_u['normalized_genre_id'] == track_v['normalized_genre_id'] else 0

    # Apply non-linear weight adjustment based on genre popularity (log scale)
    genre_weight = weights['genre'] * 1  # Keeping a simple constant weight for genre similarity

    # Criterion: Temporal proximity (time difference with logarithmic scale)
    time_diff = abs(track_u['normalized_release_date'] - track_v['normalized_release_date'])
    time_sim = 1 / (1 + math.exp(0.1 * time_diff))  # Exponential decay function

    # Adjust the time weight based on time difference using logarithmic scale
    if time_diff == 0:
        time_weight = weights['time'] * 2  # Same release date, max similarity
    else:
        time_weight = weights['time'] * math.log(time_diff + 1)  # Penalize by log of time difference

    # Criterion: Topic similarity with popularity adjustment
    topic_u = sorted([track_u[f'topic_{i + 1}'][0] for i in range(3)])
    topic_v = sorted([track_v[f'topic_{i + 1}'][0] for i in range(3)])
    topic_intersection = len(set(topic_u).intersection(set(topic_v)))
    topic_sim = topic_intersection / (len(topic_u) + len(topic_v) - topic_intersection)  # Jaccard-like similarity

    # Adjust the topic weight based on topic intersection
    topic_weight = weights['topic'] * 1  # Constant weight for topic similarity

    # Criterion: Feature similarity with a weighted intersection
    feature_u = sorted([track_u[f'feature_{i + 1}'][0] for i in range(3)])
    feature_v = sorted([track_v[f'feature_{i + 1}'][0] for i in range(3)])
    feature_intersection = len(set(feature_u).intersection(set(feature_v)))
    feature_sim = feature_intersection / (
                len(feature_u) + len(feature_v) - feature_intersection)  # Jaccard-like similarity

    # Adjust feature weight based on specificity
    feature_weight = weights['feature'] * math.sqrt(feature_intersection + 1)  # Square root to reduce impact

    # Weighted sum of all similarities with dynamic weight adjustments
    weight = (
            artist_weight * artist_sim +
            genre_weight * genre_sim +
            time_weight * time_sim +
            topic_weight * topic_sim +
            feature_weight * feature_sim
    )

    return weight


# Define weights for each criterion (can be adjusted to prioritize one criterion over another)
weights = {
    'artist': 3,
    'genre': 5,
    'time': 1,
    'topic': 4,
    'feature': 4
}

# Load the vectorized track data
file_path = 'data/vectorized_tracks.csv'
tracks_data = pd.read_csv(file_path)

# Keep only the relevant columns
columns_to_use = [
    'normalized_artist_id', 'normalized_genre_id', 'normalized_release_date',
    'topic_1', 'topic_2', 'topic_3',
    'feature_1', 'feature_2', 'feature_3',
    'track_id'
]
tracks_data = tracks_data[columns_to_use]


# Function to recommend songs
def recommend_songs(query_track_id, tracks_data, weights, exclude_ids=None, top_n=3):
    """
    Recommends similar songs based on a given track_id.

    Parameters:
    - query_track_id: ID of the reference track.
    - tracks_data: DataFrame containing the track data.
    - weights: Dictionary of weights for criteria.
    - exclude_ids: Set of track IDs to exclude.
    - top_n: Number of recommendations to return.

    Returns:
    - A list of track IDs for the recommended songs.
    """
    if exclude_ids is None:
        exclude_ids = set()

    query_track = tracks_data[tracks_data['track_id'] == query_track_id].iloc[0].to_dict()

    recommendations = []
    for _, candidate_track in tracks_data.iterrows():
        candidate_track = candidate_track.to_dict()
        if candidate_track['track_id'] in exclude_ids or candidate_track['track_id'] == query_track_id:
            continue
        weight = calculate_edge_weight(query_track, candidate_track, weights)
        recommendations.append((candidate_track['track_id'], weight))

    # Sort by descending weight and limit to top_n
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    return [rec[0] for rec in recommendations]


# Load the metadata file containing track names and artist names
metadata_file_path = 'data/tcc_ceds_music.csv'
metadata = pd.read_csv(metadata_file_path)


# Function to fetch artist and song names
def get_track_metadata(track_id, metadata):
    """
    Retrieves the artist and song name for a given track_id.

    Parameters:
    - track_id: The ID of the track.
    - metadata: DataFrame containing metadata with 'track_id', 'artist_name', and 'track_name'.

    Returns:
    - A tuple (artist_name, track_name) if the track_id exists, else (None, None).
    """
    track_info = metadata[metadata['track_id'] == track_id]
    if not track_info.empty:
        artist_name = track_info.iloc[0]['artist_name']
        track_name = track_info.iloc[0]['track_name']
        return artist_name, track_name
    return None, None


# Updated hybrid recommendation system to generate a full playlist from the original song
def hybrid_recommendation_system(initial_query_id, tracks_data, weights, metadata, max_playlist_size=100):
    """
    Hybrid system that generates a playlist starting from the initial song without recalculating at each step.
    The playlist is created by calculating the weights once for the initial song and then adding recommended tracks.

    Parameters:
    - initial_query_id: ID of the starting track.
    - tracks_data: DataFrame of track data.
    - weights: Dictionary of weights for criteria.
    - metadata: DataFrame containing track metadata.
    - max_playlist_size: Maximum size of the generated playlist.

    Returns:
    - A final playlist of recommended songs with metadata.
    """
    exclude_ids = set()
    playlist = []

    # Get the recommendations for the current track
    recommendations_with_weights = []
    query_track = tracks_data[tracks_data['track_id'] == initial_query_id].iloc[0].to_dict()

    for _, candidate_track in tracks_data.iterrows():
        candidate_track = candidate_track.to_dict()
        if candidate_track['track_id'] in exclude_ids or candidate_track['track_id'] == initial_query_id:
            continue
        weight = calculate_edge_weight(query_track, candidate_track, weights)
        recommendations_with_weights.append((candidate_track['track_id'], weight))

    # Sort recommendations by descending weight
    recommendations_with_weights = sorted(recommendations_with_weights, key=lambda x: x[1], reverse=True)

    # Add the top recommendations to the playlist until we reach the desired size
    while len(playlist) < max_playlist_size:
        # Take the top recommendation
        top_recommendation = recommendations_with_weights.pop(0)[0]  # Get track ID of the top recommendation
        playlist.append(top_recommendation)
        exclude_ids.add(top_recommendation)

        # If there are more recommendations, process the next batch
        if recommendations_with_weights:
            recommendations_with_weights = sorted(recommendations_with_weights, key=lambda x: x[1], reverse=True)
        else:
            print("No more recommendations available.")
            break  # Stop if there are no more recommendations

    # Fetch metadata for the initial track
    initial_artist, initial_song = get_track_metadata(initial_query_id, metadata)

    # Display initial song details
    if initial_artist and initial_song:
        print(f"\n--- Playlist generated starting from: ---")
        print(f"Track ID: {initial_query_id} | Artist: {initial_artist} | Song: {initial_song}")
    else:
        print(f"\n--- Playlist generated starting from unknown song with Track ID: {initial_query_id} ---")

    # Fetch metadata for the playlist
    final_playlist = []
    print(f"\n--- Final Generated Playlist ---")
    for track_id in playlist[:max_playlist_size]:
        artist_name, track_name = get_track_metadata(track_id, metadata)
        genre = metadata.loc[metadata['track_id'] == track_id, 'genre'].values[0]  # Get genre
        release_date = metadata.loc[metadata['track_id'] == track_id, 'release_date'].values[0]  # Get release date

        if artist_name and track_name:
            print(
                f"Track ID: {track_id} | Artist: {artist_name} | Song: {track_name} | Genre: {genre} | Release Date: {release_date}")
            final_playlist.append((track_id, artist_name, track_name, genre,
                                   release_date))  # Add genre and release date to the final playlist
        else:
            print(f"Track ID: {track_id} | Metadata not found!")
            final_playlist.append(
                (track_id, None, None, None, None))  # Append None for genre and release date if metadata is missing

    return final_playlist


# Example usage
initial_track_id = 8356  # Starting track ID
recommended_playlist = hybrid_recommendation_system(
    initial_query_id=initial_track_id,
    tracks_data=tracks_data,
    weights=weights,
    metadata=metadata,
    max_playlist_size=100
)
