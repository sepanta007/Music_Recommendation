# Music Recommendation System

This project is a song recommendation system that generates personalized playlists based on song similarities. It processes raw music data, extracts relevant features, and computes recommendations using structured datasets.

* [Project Structure](#project-structure)  
* [Setup & Usage](#setup--usage)
* [Features](#features)  
* [Detailed Documentation](#detailed-documentation)  
* [License](#license)  

## Project Structure

The project is organized into two main directories:

### 📂 `src/` - Source Code  
This directory contains the Python scripts necessary for data processing and recommendation generation:  
- **`preparation.py`**: Processes raw data files to generate structured datasets for the recommendation system.  
- **`recommendation.py`**: Implements the recommendation algorithm, using processed data to compute song similarities and generate playlists.  

### 📂 `data/` - Dataset  
Contains the raw and processed datasets used by the system:  
- **`music_1950_2019.csv`**: A dataset containing songs and artist information from 1950 to 2019.  
- **`data_w_genres.csv`**: Provides additional genre-related data to enrich similarity computations.  

After running `preparation.py`, three new processed files are created:  
- **`genres_with_ids.csv`**: A list of unique genres and subgenres with assigned IDs.  
- **`artist_genre_mapping.csv`**: Maps artists to their respective genres based on available data.  
- **`final_processed_music_data.csv`**: The main dataset used for recommendations, enriched with unique song and artist IDs, genre mappings, and additional metadata.  

## Setup & Usage

1. Ensure that `pandas` is installed:  
   ```bash
   pip install pandas

2. Run the data preparation script:
    ```bash
    python src/preparation.py

3. Execute the recommendation system:
   ```bash
   python src/recommendation.py

You will be prompted to enter a `track_id` from `final_processed_music_data.csv` to generate recommendations.

4. The output will be stored as:

   ```bash
   data/generated_playlist_from_<song_name>.csv

This file contains the top 100 recommended songs, ranked by similarity score.

## Features

- **Data-Driven Recommendations**: Uses structured data to improve playlist quality.  
- **Genre-Based Similarity**: Enhances recommendations by leveraging genre mappings.  
- **Top 100 Playlist Generation**: Provides a ranked list of the most similar songs.  
- **Efficient Processing**: Prepares and optimizes datasets for better performance.

## Detailed Documentation

For a more detailed explanation of the methodology, data processing, and recommendation algorithm, refer to the report [Music_Recommendation.pdf](https://github.com/sepanta007/Music_Recommendation/blob/master/Music_Recommendation.pdf) included in this repository. 

## License

This project is licensed under the [MIT](https://github.com/sepanta007/Music_Recommendation/blob/master/LICENSE) License.  
