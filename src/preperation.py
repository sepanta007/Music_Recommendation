import pandas as pd

# Load the dataset
file_path = 'data/tcc_ceds_music.csv'
music_data = pd.read_csv(file_path)

# Assign unique IDs to each unique (track_name, artist_name) combination
music_data['track_id'] = music_data[['track_name', 'artist_name']].apply(tuple, axis=1).factorize()[0] + 1

# Save the modified DataFrame back to the CSV file
music_data.to_csv(file_path, index=False)  # Overwrites the original file

print("The track_id column has been added to the CSV file, considering both track_name and artist_name.")

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

# Save the modified DataFrame back to the CSV file
music_data.to_csv(file_path, index=False)  # Overwrites the original file

print("The genre_id column has been added to the CSV file based on the specified genre mapping.")