import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add src to sys.path to allow pickler to find collaborative_filtering directly
sys.path.append(str(Path(__file__).parent))

class RecommendationEngine:
    def __init__(self, data_dir="data/processed", models_dir="models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Load Data if present, otherwise fallback to precomputed dashboard stats
        movies_path = self.data_dir / "movies_master.parquet"
        ratings_path = self.data_dir / "ratings_clean.parquet"
        
        if movies_path.exists() and ratings_path.exists():
            self.movies = pd.read_parquet(movies_path)
            self.ratings = pd.read_parquet(ratings_path)
            self._compute_popular_movies()
        else:
            print("Notice: Parquet datasets not found in data/processed, loading fallback from models/dashboard_stats.pkl...")
            self.movies = pd.DataFrame(columns=['movieId', 'clean_title', 'genres', 'imdbId'])
            self.ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
            self.popular_movies = pd.DataFrame(columns=['movieId', 'score'])
            
            stats_path = self.models_dir / "dashboard_stats.pkl"
            if stats_path.exists():
                try:
                    stats = joblib.load(stats_path)
                    top_df = stats.get('top_rated', pd.DataFrame())
                    if not top_df.empty:
                        rows = []
                        for idx, row in top_df.iterrows():
                            title = row.get('Movie', f'Movie {idx}')
                            avg_r = float(row.get('Avg Rating', 4.0))
                            rows.append({
                                'movieId': int(idx + 1),
                                'clean_title': str(title),
                                'genres': 'Drama, Thriller',
                                'score': round(avg_r, 2),
                                'imdbId': None
                            })
                        fallback_df = pd.DataFrame(rows)
                        self.movies = fallback_df
                        self.popular_movies = fallback_df[['movieId', 'score']]
                except Exception as e:
                    print(f"Warning loading fallback stats: {e}")
        
        # Models will be lazily loaded via properties
        self._svd_model = None
        self._item_cf_model = None
        self._user_cf_model = None

    @property
    def svd_model(self):
        if self._svd_model is None:
            try:
                self._svd_model = joblib.load(self.models_dir / "svd_model.pkl")
            except FileNotFoundError:
                print("Warning: svd_model.pkl not found.")
        return self._svd_model

    @property
    def item_cf_model(self):
        if self._item_cf_model is None:
            try:
                self._item_cf_model = joblib.load(self.models_dir / "item_cf_model.pkl")
            except FileNotFoundError:
                print("Warning: item_cf_model.pkl not found.")
        return self._item_cf_model

    @property
    def user_cf_model(self):
        if self._user_cf_model is None:
            try:
                self._user_cf_model = joblib.load(self.models_dir / "user_cf_model.pkl")
            except FileNotFoundError:
                print("Warning: user_cf_model.pkl not found.")
        return self._user_cf_model

    def _compute_popular_movies(self):
        """Precomputes a weighted rating for popular movies."""
        movie_stats = self.ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['mean_rating', 'rating_count']
        
        # Bayesian average (IMDB formula)
        m = 500  # Minimum ratings required to be listed in top 250
        C = movie_stats['mean_rating'].mean() # Mean rating across the whole report
        
        q_movies = movie_stats.copy()
        
        def weighted_rating(x, m=m, C=C):
            v = x['rating_count']
            R = x['mean_rating']
            return (v/(v+m) * R) + (m/(m+v) * C)
            
        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        self.popular_movies = q_movies.sort_values('score', ascending=False)

    def top_n_popular(self, n=10, genre=None):
        """Popularity-based fallback for cold-start users."""
        if 'clean_title' in self.popular_movies.columns:
            pop_movies = self.popular_movies
        else:
            cols_to_merge = [c for c in self.movies.columns if c == 'movieId' or c not in self.popular_movies.columns]
            pop_movies = self.popular_movies.merge(self.movies[cols_to_merge], on='movieId')
            
        if genre and 'genres' in pop_movies.columns:
            pop_movies = pop_movies[pop_movies['genres'].str.contains(genre, case=False, na=False)]
        
        cols = [c for c in ['movieId', 'clean_title', 'genres', 'score', 'imdbId'] if c in pop_movies.columns]
        return pop_movies[cols].head(n).to_dict('records')

    def recommend_for_user(self, user_id, n=10):
        """Uses SVD to recommend movies for a user. Handles cold-start."""
        if self.svd_model is None:
            return self.top_n_popular(n)
            
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        # Cold start handling: if user has < 5 ratings, fallback to popular
        if len(user_ratings) < 5:
            return self.top_n_popular(n)
            
        rated_movie_ids = set(user_ratings['movieId'].values)
        all_movie_ids = self.movies['movieId'].unique()
        
        unrated_movie_ids = [m for m in all_movie_ids if m not in rated_movie_ids]
        
        # Predict ratings for all unrated movies
        predictions = []
        for movie_id in unrated_movie_ids:
            pred = self.svd_model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
            
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n]
        
        results = []
        for movie_id, est_rating in top_predictions:
            m_match = self.movies[self.movies['movieId'] == movie_id]
            if not m_match.empty:
                movie_info = m_match.iloc[0]
                clean_title = str(movie_info.get('clean_title', f"Movie {movie_id}"))
                genres = str(movie_info.get('genres', 'Drama'))
                imdb_id = str(movie_info.get('imdbId', '')) if pd.notna(movie_info.get('imdbId', None)) else None
            else:
                clean_title = f"Movie {movie_id}"
                genres = "Drama, Thriller"
                imdb_id = None
                
            results.append({
                'movieId': int(movie_id),
                'clean_title': clean_title,
                'genres': genres,
                'imdbId': imdb_id,
                'predicted_rating': round(float(est_rating), 2)
            })
            
        return results

    def recommend_similar_movies(self, movie_title, n=10):
        """Finds similar movies using Item-based CF and fuzzy title matching."""
        if self.item_cf_model is None:
            return {"query_movie": movie_title, "recommendations": []}
            
        # Fuzzy match title
        matches = self.movies[self.movies['clean_title'].str.contains(movie_title, case=False, na=False)]
        if len(matches) == 0:
            return {"query_movie": movie_title, "recommendations": []}
            
        # Take the most popular match if there are multiple
        movie_id = matches.iloc[0]['movieId']
        actual_title = matches.iloc[0]['clean_title']
        
        similar_items = self.item_cf_model.get_similar_items(movie_id, n)
        
        results = []
        for sim_movie_id, similarity in similar_items:
            m_match = self.movies[self.movies['movieId'] == sim_movie_id]
            if not m_match.empty:
                movie_info = m_match.iloc[0]
                clean_title = str(movie_info.get('clean_title', f"Movie {sim_movie_id}"))
                genres = str(movie_info.get('genres', 'Drama'))
                imdb_id = str(movie_info.get('imdbId', '')) if pd.notna(movie_info.get('imdbId', None)) else None
            else:
                clean_title = f"Movie {sim_movie_id}"
                genres = "Drama, Thriller"
                imdb_id = None
                
            results.append({
                'movieId': int(sim_movie_id),
                'clean_title': clean_title,
                'genres': genres,
                'imdbId': imdb_id,
                'similarity_score': round(float(similarity), 4)
            })
            
        return {"query_movie": actual_title, "recommendations": results}

if __name__ == "__main__":
    engine = RecommendationEngine()
    print("Testing Popularity:")
    print(engine.top_n_popular(n=3))
    
    print("\nTesting User Recs (Cold Start):")
    print(engine.recommend_for_user(user_id=9999999, n=3)) # Unknown user
    
    print("\nTesting Similar Movies:")
    print(engine.recommend_similar_movies("Toy Story", n=3))
