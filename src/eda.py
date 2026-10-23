import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
import sys

def perform_eda():
    processed_dir = Path("data/processed")
    figures_dir = processed_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw/ml-20m")
    
    print("Loading data...")
    movies = pd.read_parquet(processed_dir / "movies_master.parquet")
    ratings = pd.read_parquet(processed_dir / "ratings_clean.parquet")
    
    # 1. Rating Distribution
    print("Plotting rating distribution...")
    plt.figure(figsize=(10, 6))
    rating_counts = ratings['rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(figures_dir / "rating_distribution.png", dpi=300)
    plt.close()
    
    # 2. Most Rated Movies
    print("Plotting most rated movies...")
    movie_counts = ratings['movieId'].value_counts()
    top_n_counts = movie_counts.head(20).to_frame(name='count').merge(movies, on='movieId')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='clean_title', data=top_n_counts, palette='mako')
    plt.title('Top 20 Most Rated Movies')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movie Title')
    plt.tight_layout()
    plt.savefig(figures_dir / "most_rated_movies.png", dpi=300)
    plt.close()
    
    # 3. Highest Rated Movies (min 1000 ratings)
    print("Plotting highest rated movies...")
    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['mean_rating', 'rating_count']
    high_rated = movie_stats[movie_stats['rating_count'] >= 1000].sort_values('mean_rating', ascending=False).head(20)
    high_rated = high_rated.reset_index().merge(movies, on='movieId')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mean_rating', y='clean_title', data=high_rated, palette='rocket')
    plt.title('Top 20 Highest Rated Movies (Min 1000 Ratings)')
    plt.xlabel('Average Rating')
    plt.ylabel('Movie Title')
    plt.xlim(high_rated['mean_rating'].min() - 0.2, 5.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "highest_rated_movies.png", dpi=300)
    plt.close()
    
    # 4. Genre Breakdown
    print("Plotting genre breakdown...")
    genres_expanded = movies['genres'].str.split(',', expand=True).stack().reset_index(level=1, drop=True)
    genres_expanded = genres_expanded[genres_expanded != '(no genres listed)']
    genre_counts = genres_expanded.value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='flare')
    plt.title('Number of Movies per Genre')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(figures_dir / "genre_breakdown.png", dpi=300)
    plt.close()

    # 5. User activity distribution
    print("Plotting user activity distribution...")
    user_counts = ratings['userId'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts[user_counts <= 500], bins=50, kde=False, color='coral')
    plt.title('User Activity Distribution (capped at 500 ratings for visibility)')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig(figures_dir / "user_activity.png", dpi=300)
    plt.close()

    # 6. Process and analyze tags
    print("Processing tags...")
    try:
        tags = pd.read_csv(raw_dir / "tags.csv")
        tags['tag'] = tags['tag'].astype(str).str.lower().str.strip()
        unique_tags = tags['tag'].nunique()
        movies_with_tags = tags['movieId'].nunique()
        movies_total = movies['movieId'].nunique()
        movies_without_tags = movies_total - movies_with_tags
        
        # Save processed tags for phase 7
        tags_clean = tags.drop_duplicates(subset=['userId', 'movieId', 'tag'])
        tags_clean.to_parquet(processed_dir / "tags_clean.parquet", index=False)
        
        print(f"Unique Tags: {unique_tags}")
        print(f"Movies with tags: {movies_with_tags}")
        print(f"Movies without tags: {movies_without_tags}")
        
        # Write tags summary to text file for reading back
        with open(processed_dir / "tags_summary.txt", "w") as f:
            f.write(f"{unique_tags},{movies_with_tags},{movies_without_tags}\n")

    except Exception as e:
        print(f"Error processing tags: {e}")

if __name__ == "__main__":
    perform_eda()
