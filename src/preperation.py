import pandas as pd

# Load the dataset
file_path = 'data/tcc_ceds_music.csv'
music_data = pd.read_csv(file_path)

# Assign unique IDs to each unique (track_name, artist_name) combination
music_data['track_id'] = music_data[['track_name', 'artist_name']].apply(tuple, axis=1).factorize()[0] + 1

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

# Assign unique IDs to each unique artist_name
music_data['artist_id'] = music_data['artist_name'].factorize()[0] + 1

# Save the modified DataFrame back to the CSV file (single save to optimize)
music_data.to_csv(file_path, index=False)  # Overwrites the original file

print("The track_id, genre_id, and artist_id columns have been added to the CSV file.")
