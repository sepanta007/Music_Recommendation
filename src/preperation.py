import pandas as pd
import ast
import os


def validate_file(file_path):
    """Check if the file exists and is readable."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def generate_unique_ID_for_each_genre():
    """
    Generates a unique ID for each genre in the 'genres' column of the dataset.
    Saves the mapping to 'genres_with_ids.csv'.
    """
    input_file = 'data/data_w_genres.csv'
    output_file = 'data/genres_with_ids.csv'

    # Validate file existence
    validate_file(input_file)

    # Read the CSV file
    data = pd.read_csv(input_file)

    # Extract all genres from the 'genres' column
    genres = data['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    unique_genres = set(genre for sublist in genres for genre in sublist)

    # Create a mapping of genres to unique IDs
    genre_to_id = {genre: idx for idx, genre in enumerate(sorted(unique_genres), start=1)}

    # Save the mapping to a new CSV file
    genre_mapping = pd.DataFrame(list(genre_to_id.items()), columns=['Genre', 'ID'])
    genre_mapping.to_csv(output_file, index=False)
    print(f"Genre-ID mapping saved to {output_file}")


def generate_artist_genre_mapping():
    """
    Maps artists to their associated genre IDs.
    Saves the mapping to 'artist_genre_mapping.csv'.
    """
    input_file = 'data/data_w_genres.csv'
    output_file = 'data/artist_genre_mapping.csv'
    artist_column = 'artists'
    genre_column = 'genres'

    # Validate file existence
    validate_file(input_file)

    # Load the dataset
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    if artist_column not in df.columns or genre_column not in df.columns:
        raise ValueError(f"Columns '{artist_column}' or '{genre_column}' not found in the dataset!")

    # Generate a unique genre-to-ID mapping
    unique_genres = set()
    df[genre_column].dropna().apply(
        lambda x: unique_genres.update(ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    )
    genre_to_id = {genre: idx + 1 for idx, genre in enumerate(sorted(unique_genres))}

    # Map each artist to their associated genre IDs
    artist_genres = {}
    for _, row in df.iterrows():
        artist_name = row[artist_column]
        genres = ast.literal_eval(row[genre_column]) if pd.notna(row[genre_column]) and row[genre_column].strip() else []
        genre_ids = {genre_to_id[genre] for genre in genres} if genres else set()
        if artist_name in artist_genres:
            artist_genres[artist_name].update(genre_ids)
        else:
            artist_genres[artist_name] = genre_ids

    # Convert artist_genres dictionary into a DataFrame
    artist_genre_df = pd.DataFrame(
        [(artist, list(genre_ids)) for artist, genre_ids in artist_genres.items()],
        columns=['Artist', 'Genre_IDs']
    )

    # Save the DataFrame to a CSV file
    artist_genre_df.to_csv(output_file, index=False)
    print(f"Artist-genre mapping saved to {output_file}")


def process_music_data(file_path, mapping_file, output_file):
    """
    Processes a music dataset by:
    1. Dropping specified columns.
    2. Adding 'artist_id' and 'track_id' columns.
    3. Updating the 'genre' column based on artist_genre_mapping.csv.
    4. Computing 'topic_1', 'topic_2', and 'topic_3' for top topics.
    5. Computing 'feature_1', 'feature_2', and 'feature_3' for top features.
    6. Dropping rows with null values and saving the result.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - mapping_file (str): Path to the artist_genre_mapping.csv file.
    - output_file (str): Path to save the processed CSV file.
    """
    # Validate file existence
    validate_file(file_path)
    validate_file(mapping_file)

    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop specified columns
    columns_to_drop = ['Unnamed: 0', 'lyrics', 'age', 'len', 'topic']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop null values
    df.replace('<unset>', pd.NA, inplace=True)
    df = df.dropna()

    # Assign unique IDs for artist_name
    artist_id_map = {artist: i + 1 for i, artist in enumerate(df['artist_name'].unique())}
    df.insert(df.columns.get_loc('artist_name') + 1, 'artist_id', df['artist_name'].map(artist_id_map))

    # Assign unique IDs for each track
    df.insert(df.columns.get_loc('track_name') + 1, 'track_id', range(1, len(df) + 1))

    # Load the artist-genre mapping
    mapping_df = pd.read_csv(mapping_file)
    mapping_df['Artist'] = mapping_df['Artist'].str.lower()
    df['artist_name'] = df['artist_name'].str.lower()
    artist_genre_map = dict(zip(mapping_df['Artist'], mapping_df['Genre_IDs']))

    # Define default genre mappings
    default_genre_map = {
        'rock': [2340],
        'blues': [300],
        'pop': [2170],
        'country': [698],
        'jazz': [1527],
        'reggae': [2316],
        'hip hop' : [1292]
    }

    # Function to update genre
    def update_genre(row):
        artist = row['artist_name']
        genre = row['genre'].lower() if pd.notna(row['genre']) else None
        if artist in artist_genre_map:
            if artist_genre_map[artist] != "[]":
                return artist_genre_map[artist]
            else :
                return default_genre_map[genre]
        elif genre in default_genre_map:
            return default_genre_map[genre]
        else:
            return None

    df['genre'] = df.apply(update_genre, axis=1)

    # Process topics and features
    topic_columns = [
        'dating', 'violence', 'world/life', 'night/time', 'shake the audience', 'family/gospel','romantic',
        'communication', 'obscene', 'music', 'movement/places','light/visual perceptions', 'family/spiritual',
        'like/girls', 'sadness', 'feelings'
    ]
    feature_columns = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy']

    def get_top_values(row, columns):
        values = [(i + 1, row[col]) for i, col in enumerate(columns)]
        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)[:3]
        return sorted_values + [(None, None)] * (3 - len(sorted_values))

    # Compute top 3 topics and features
    df[['topic_1', 'topic_2', 'topic_3']] = df.apply(
        lambda row: pd.Series(get_top_values(row, topic_columns)), axis=1
    )
    df[['feature_1', 'feature_2', 'feature_3']] = df.apply(
        lambda row: pd.Series(get_top_values(row, feature_columns)), axis=1
    )

    # Drop original topic and feature columns
    df = df.drop(columns=topic_columns + feature_columns, errors='ignore')

    # Save the processed DataFrame
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


# Execute all steps
generate_unique_ID_for_each_genre()
generate_artist_genre_mapping()
process_music_data('data/music_1950_2019.csv',
                   'data/artist_genre_mapping.csv', 'data/final_processed_music_data.csv')