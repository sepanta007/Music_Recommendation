import pandas as pd


# Function to calculate the similarity weight between two tracks
def calculate_edge_weight(track_u, track_v, weights):
    """
    Calculates the weighted similarity between two tracks based on multiple criteria.

    Parameters:
    - track_u, track_v: Dictionaries representing the tracks.
    - weights: Dictionary of weights for each criterion.

    Returns:
    - The weighted similarity value.
    """
    # Criterion: Artist similarity
    artist_sim = 1 if track_u['normalized_artist_id'] == track_v['normalized_artist_id'] else 0

    # Criterion: Genre similarity
    genre_sim = 1 if track_u['normalized_genre_id'] == track_v['normalized_genre_id'] else 0

    # Criterion: Temporal proximity (release year)
    time_diff = abs(track_u['normalized_release_date'] - track_v['normalized_release_date'])
    time_sim = 1 / (1 + time_diff)

    # Criterion: Similarity of dominant topics
    topic_sim = sum(
        (3 - i) * (track_u[f'topic_{i + 1}'] == track_v[f'topic_{i + 1}'])
        for i in range(3)
    )

    # Criterion: Similarity of dominant audio features
    feature_sim = sum(
        (3 - i) * (track_u[f'feature_{i + 1}'] == track_v[f'feature_{i + 1}'])
        for i in range(3)
    )

    # Weighted linear combination
    weight = (
        weights['artist'] * artist_sim +
        weights['genre'] * genre_sim +
        weights['time'] * time_sim +
        weights['topic'] * topic_sim +
        weights['feature'] * feature_sim
    )

    return weight


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

# Define weights for each criterion
weights = {
    'artist': 5,
    'genre': 3,
    'time': 1,
    'topic': 2,
    'feature': 2
}


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


# Updated hybrid recommendation system with final playlist details
def hybrid_recommendation_system(initial_query_id, tracks_data, weights, metadata, interactions=3,
                                 max_playlist_size=100):
    """
    Hybrid system that initially recommends a list of songs
    and recalculates after a set number of interactions.

    Parameters:
    - initial_query_id: ID of the starting track.
    - tracks_data: DataFrame of track data.
    - weights: Dictionary of weights for criteria.
    - metadata: DataFrame containing track metadata.
    - interactions: Number of interactions before recalculating recommendations.
    - max_playlist_size: Maximum size of the generated playlist.

    Returns:
    - A final playlist of recommended songs with metadata.
    """
    current_query_id = initial_query_id
    exclude_ids = set()
    playlist = []

    while len(playlist) < max_playlist_size:
        print(f"\n--- Calculating recommendations for track ID {current_query_id} ---")

        # Get the top 3 recommendations for the current track
        recommendations_with_weights = []
        query_track = tracks_data[tracks_data['track_id'] == current_query_id].iloc[0].to_dict()

        for _, candidate_track in tracks_data.iterrows():
            candidate_track = candidate_track.to_dict()
            if candidate_track['track_id'] in exclude_ids or candidate_track['track_id'] == current_query_id:
                continue
            weight = calculate_edge_weight(query_track, candidate_track, weights)
            recommendations_with_weights.append((candidate_track['track_id'], weight))

        # Sort recommendations by descending weight
        recommendations_with_weights = sorted(recommendations_with_weights, key=lambda x: x[1], reverse=True)

        # Display all candidate recommendations with weights
        print( # Display top 10 for brevity
            f"First 10 candidates between all candidates with weights (sorted): {recommendations_with_weights[:10]}")

        # Take the top 3 recommendations
        new_recommendations = [rec[0] for rec in recommendations_with_weights[:3]]
        print(f"Selected top 3 recommendations: {new_recommendations}")

        # Add new recommendations to the playlist
        playlist.extend(new_recommendations)
        exclude_ids.update(new_recommendations)

        if len(new_recommendations) == 0:
            print("No more recommendations available. Stopping.")
            break  # Stop if no more recommendations are available

        # Move to the third track in the recommendations if available
        current_query_id = new_recommendations[min(2, len(new_recommendations) - 1)]
        print(f"Next query track set to: {current_query_id}")

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
        if artist_name and track_name:
            print(f"Track ID: {track_id} | Artist: {artist_name} | Song: {track_name}")
            final_playlist.append((track_id, artist_name, track_name))
        else:
            print(f"Track ID: {track_id} | Metadata not found!")
            final_playlist.append((track_id, None, None))

    return final_playlist


# Example usage
initial_track_id = 1  # Starting track ID
recommended_playlist = hybrid_recommendation_system(
    initial_query_id=initial_track_id,
    tracks_data=tracks_data,
    weights=weights,
    metadata=metadata,
    interactions=3,
    max_playlist_size=100
)
