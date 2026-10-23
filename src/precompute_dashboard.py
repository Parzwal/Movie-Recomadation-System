import pandas as pd
import joblib
from pathlib import Path

def main():
    print("Loading data...")
    data_dir = Path("data/processed")
    models_dir = Path("models")
    
    movies = pd.read_parquet(data_dir / "movies_master.parquet")
    ratings = pd.read_parquet(data_dir / "ratings_clean.parquet")
    
    print("Computing statistics on 20M ratings...")
    # 1. Global Metrics
    total_movies = len(movies)
    total_ratings = len(ratings)
    avg_rating = ratings['rating'].mean()
    
    # 2. Distributions
    rating_counts = ratings['rating'].value_counts().sort_index()
    user_counts = ratings['userId'].value_counts()
    
    # 3. Top / Most Rated
    stats = ratings.groupby('movieId').agg(mean_rating=('rating', 'mean'), count=('rating', 'count'))
    
    top_rated = stats[stats['count'] >= 5000].sort_values('mean_rating', ascending=False).head(10)
    top_rated = top_rated.merge(movies, on='movieId')[['clean_title', 'mean_rating', 'count']]
    top_rated.columns = ['Movie', 'Avg Rating', 'Total Ratings']
    
    most_rated = stats.sort_values('count', ascending=False).head(10)
    most_rated = most_rated.merge(movies, on='movieId')[['clean_title', 'count', 'mean_rating']]
    most_rated.columns = ['Movie', 'Total Ratings', 'Avg Rating']
    
    # Assemble final stats dict
    dashboard_stats = {
        'total_movies': total_movies,
        'total_ratings': total_ratings,
        'avg_rating': float(avg_rating),
        'rating_counts': rating_counts,
        'user_counts': user_counts,
        'top_rated': top_rated,
        'most_rated': most_rated
    }
    
    out_path = models_dir / "dashboard_stats.pkl"
    print(f"Saving to {out_path}...")
    joblib.dump(dashboard_stats, out_path)
    print("Done! Dashboard stats precomputed successfully.")

if __name__ == "__main__":
    main()
