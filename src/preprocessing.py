import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_year(title):
    """Extracts year from movie title (e.g. 'Toy Story (1995)')"""
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        return int(match.group(1))
    return np.nan

def clean_title(title):
    """Removes the year from the title and fixes article formatting."""
    t = re.sub(r'\s*\(\d{4}\)\s*$', '', str(title))
    
    # Fix trailing articles: "Matrix, The" -> "The Matrix", "Short Film About Killing, A" -> "A Short Film About Killing"
    match = re.search(r'^(.*?),\s+(The|A|An|La|El|Los|Las|Le|Les)\s*$', t, flags=re.IGNORECASE)
    if match:
        # Move the article to the front
        t = f"{match.group(2)} {match.group(1)}"
        
    return t.strip()

def process_data(raw_data_dir: Path, processed_data_dir: Path):
    """
    Loads raw CSVs, cleans and merges data, and exports to parquet.
    Generates summary statistics.
    """
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify which dataset was downloaded (ml-20m or ml-latest-small)
    if (raw_data_dir / "ml-20m").exists():
        dataset_path = raw_data_dir / "ml-20m"
        print("Using ml-20m dataset")
    elif (raw_data_dir / "ml-latest-small").exists():
        dataset_path = raw_data_dir / "ml-latest-small"
        print("Using ml-latest-small dataset")
    else:
        raise FileNotFoundError("Could not find extracted MovieLens dataset in data/raw/")
        
    print("Loading movies...")
    movies_df = pd.read_csv(dataset_path / "movies.csv")
    print("Loading ratings...")
    ratings_df = pd.read_csv(dataset_path / "ratings.csv")
    
    print("Processing movies...")
    movies_df['year'] = movies_df['title'].apply(extract_year)
    movies_df['clean_title'] = movies_df['title'].apply(clean_title)
    movies_df['genres'] = movies_df['genres'].str.replace('|', ',', regex=False)
    
    print("Processing ratings...")
    # Drop duplicates if any (user rating same movie twice)
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='last')
    
    # For tags and links, we could also load them if needed later, but they are not strictly required for the core engine.
    # We will load links to join TMDB ids.
    print("Loading links...")
    links_df = pd.read_csv(dataset_path / "links.csv")
    movies_master = movies_df.merge(links_df, on='movieId', how='left')
    
    # Save to parquet
    print("Saving processed files to parquet...")
    movies_master.to_parquet(processed_data_dir / "movies_master.parquet", index=False)
    ratings_df.to_parquet(processed_data_dir / "ratings_clean.parquet", index=False)
    
    # Generate Summary Statistics
    n_users = ratings_df['userId'].nunique()
    n_movies = ratings_df['movieId'].nunique()
    n_ratings = len(ratings_df)
    sparsity = 1.0 - (n_ratings / (n_users * n_movies))
    
    ratings_per_user = ratings_df.groupby('userId').size()
    ratings_per_movie = ratings_df.groupby('movieId').size()
    
    print("\n--- Summary Statistics ---")
    print(f"Total Users: {n_users:,}")
    print(f"Total Movies: {n_movies:,}")
    print(f"Total Ratings: {n_ratings:,}")
    print(f"Matrix Sparsity: {sparsity:.4%}")
    
    print("\nRatings per User:")
    print(f"  Min: {ratings_per_user.min()}")
    print(f"  Max: {ratings_per_user.max()}")
    print(f"  Median: {ratings_per_user.median()}")
    print(f"  Mean: {ratings_per_user.mean():.2f}")
    
    print("\nRatings per Movie:")
    print(f"  Min: {ratings_per_movie.min()}")
    print(f"  Max: {ratings_per_movie.max()}")
    print(f"  Median: {ratings_per_movie.median()}")
    print(f"  Mean: {ratings_per_movie.mean():.2f}")
    
    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "ratings_per_user": ratings_per_user.describe().to_dict(),
        "ratings_per_movie": ratings_per_movie.describe().to_dict()
    }

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    process_data(raw_dir, processed_dir)
