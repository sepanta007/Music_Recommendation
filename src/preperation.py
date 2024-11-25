import pandas as pd

# Load the dataset
file_path = 'data/tcc_ceds_music.csv'
music_data = pd.read_csv(file_path)

# Display initial columns for verification
print("Initial columns:", music_data.columns)

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'lyrics', 'len', 'age']
music_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Save the cleaned dataset back to the same file
output_path = 'data/tcc_ceds_music.csv'
music_data.to_csv(output_path, index=False)

print(f"Dataset saved after removing columns {columns_to_drop}. Current columns:")

# Assign unique IDs to each unique (track_name, artist_name) combination
music_data['track_id'] = music_data[['track_name', 'artist_name']].apply(tuple, axis=1).factorize()[0] + 1

# Assign unique IDs to each unique artist_name
music_data['artist_id'] = music_data['artist_name'].factorize()[0] + 1

# Define genre mapping
genre_mapping = {
    'pop': 1,
    'country': 2,
    'blues': 3,
    'jazz': 4,
    'reggae': 5,
    'rock': 6,
    'hip hop': 7
}

# Map genres to their corresponding IDs
music_data['genre_id'] = music_data['genre'].map(genre_mapping)

# Define columns for topics and audio features
topic_columns = [
    'dating', 'violence', 'world/life', 'night/time', 'shake the audience',
    'family/gospel', 'romantic', 'communication', 'obscene', 'music',
    'movement/places', 'light/visual perceptions', 'family/spiritual',
    'like/girls', 'sadness', 'feelings'
]

audio_features = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy']

# Convert topic_columns and audio_features to numeric with detailed inspection
for col in topic_columns + audio_features:
    print(f"Cleaning and converting column: {col}")
    music_data[col] = pd.to_numeric(music_data[col], errors='coerce')  # Force numeric conversion
    print(f"NaN values detected in column '{col}': {music_data[col].isna().sum()}")
    music_data[col].fillna(0, inplace=True)  # Replace NaN with 0
    print(f"Unique values in '{col}': {music_data[col].unique()[:10]}")  # Display first 10 unique values

# Process each track and create vectors
vector_data = []
for _, row in music_data.iterrows():
    # Extract top 3 topics
    try:
        top_topics = row[topic_columns].astype(float).nlargest(3).index.tolist()
        topic_1, topic_2, topic_3 = top_topics
    except Exception as e:
        print(f"Error processing row: {row}")
        print(f"Exception: {e}")
        continue  # Skip problematic rows

    # Extract top 3 audio features
    try:
        top_features = row[audio_features].astype(float).nlargest(3).index.tolist()
        feature_1, feature_2, feature_3 = top_features
    except Exception as e:
        print(f"Error processing audio features for row: {row}")
        print(f"Exception: {e}")
        continue  # Skip problematic rows

    # Create a vector with required fields
    vector = {
        'track_id': row['track_id'],
        'artist_id': row['artist_id'],
        'release_date': row['release_date'],
        'genre_id': row['genre_id'],
        'topic_1': topic_1,
        'topic_2': topic_2,
        'topic_3': topic_3,
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3
    }
    vector_data.append(vector)

# Convert the list of vectors into a DataFrame
vector_df = pd.DataFrame(vector_data)

# Save the vectorized data to a CSV file
output_path = 'data/vectorized_tracks.csv'
vector_df.to_csv(output_path, index=False)

print(f"Vectorized track data has been saved to '{output_path}'.")
