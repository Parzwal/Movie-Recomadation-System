import os
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

class UserBasedCF:
    """
    User-based Collaborative Filtering using k-Nearest Neighbors.
    
    Mathematics:
    Given a target user u, we find the k most similar users based on the cosine
    similarity of their rating vectors. The similarity between user u and user v is:
    sim(u, v) = dot(r_u, r_v) / (||r_u|| * ||r_v||)
    
    To handle power users (who rate thousands of movies and become overly connected hubs),
    we cap the maximum number of ratings per user to `max_ratings_per_user` via random sampling
    before constructing the sparse matrix.
    """
    def __init__(self, k=20, max_ratings_per_user=500):
        self.k = k
        self.max_ratings_per_user = max_ratings_per_user
        self.model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        self.user_mapper = {}
        self.user_inv_mapper = {}
        self.item_mapper = {}
        self.item_inv_mapper = {}
        self.sparse_matrix = None
        
    def fit(self, ratings_df):
        print("Fitting UserBasedCF...")
        # Cap power users
        print(f"Capping user ratings to max {self.max_ratings_per_user}...")
        # Shuffle first to ensure random sampling, then take the first N per user
        shuffled = ratings_df.sample(frac=1, random_state=42)
        sampled_ratings = shuffled.groupby('userId').head(self.max_ratings_per_user)
        
        # Create mappings
        unique_users = sampled_ratings['userId'].unique()
        unique_items = sampled_ratings['movieId'].unique()
        
        self.user_mapper = dict(zip(unique_users, range(len(unique_users))))
        self.user_inv_mapper = dict(zip(range(len(unique_users)), unique_users))
        
        self.item_mapper = dict(zip(unique_items, range(len(unique_items))))
        self.item_inv_mapper = dict(zip(range(len(unique_items)), unique_items))
        
        user_idx = [self.user_mapper[i] for i in sampled_ratings['userId']]
        item_idx = [self.item_mapper[i] for i in sampled_ratings['movieId']]
        
        # Build sparse matrix (Users x Items)
        print("Building sparse matrix...")
        self.sparse_matrix = csr_matrix(
            (sampled_ratings['rating'], (user_idx, item_idx)),
            shape=(len(unique_users), len(unique_items))
        )
        
        print("Fitting KNN model...")
        self.model.fit(self.sparse_matrix)
        print("UserBasedCF fitted successfully.")
        
    def get_similar_users(self, user_id, n=10):
        if user_id not in self.user_mapper:
            return []
        
        user_idx = self.user_mapper[user_id]
        user_vec = self.sparse_matrix[user_idx]
        
        distances, indices = self.model.kneighbors(user_vec, n_neighbors=n+1)
        
        # Exclude the user themselves (index 0 is usually the user if distance is 0)
        similar_users = []
        for i in range(1, len(distances[0])):
            neighbor_idx = indices[0][i]
            neighbor_id = self.user_inv_mapper[neighbor_idx]
            similar_users.append((neighbor_id, 1 - distances[0][i])) # convert distance to similarity
            
        return similar_users


class ItemBasedCF:
    """
    Item-based Collaborative Filtering using precomputed Cosine Similarity.
    
    Mathematics:
    The similarity between item i and item j is the cosine similarity of their 
    rating vectors across all users:
    sim(i, j) = dot(r_i, r_j) / (||r_i|| * ||r_j||)
    
    Due to memory constraints with 27k items, we restrict the item-item matrix 
    to movies that have at least `min_item_ratings` to remove noise and keep the
    matrix dense enough to be useful.
    """
    def __init__(self, min_item_ratings=50):
        self.min_item_ratings = min_item_ratings
        self.item_mapper = {}
        self.item_inv_mapper = {}
        self.item_sim_matrix = None
        
    def fit(self, ratings_df):
        print(f"Fitting ItemBasedCF (min {self.min_item_ratings} ratings per movie)...")
        # Filter items with less than min ratings
        item_counts = ratings_df['movieId'].value_counts()
        popular_items = item_counts[item_counts >= self.min_item_ratings].index
        
        filtered_ratings = ratings_df[ratings_df['movieId'].isin(popular_items)]
        
        unique_users = filtered_ratings['userId'].unique()
        unique_items = filtered_ratings['movieId'].unique()
        
        self.item_mapper = dict(zip(unique_items, range(len(unique_items))))
        self.item_inv_mapper = dict(zip(range(len(unique_items)), unique_items))
        
        user_mapper = dict(zip(unique_users, range(len(unique_users))))
        
        user_idx = [user_mapper[i] for i in filtered_ratings['userId']]
        item_idx = [self.item_mapper[i] for i in filtered_ratings['movieId']]
        
        # Items x Users matrix
        sparse_matrix = csr_matrix(
            (filtered_ratings['rating'], (item_idx, user_idx)),
            shape=(len(unique_items), len(unique_users))
        )
        
        print("Computing cosine similarity matrix...")
        self.item_sim_matrix = cosine_similarity(sparse_matrix, dense_output=False)
        print("ItemBasedCF fitted successfully.")
        
    def get_similar_items(self, movie_id, n=10):
        if movie_id not in self.item_mapper:
            return []
            
        item_idx = self.item_mapper[movie_id]
        
        # Get the row from the similarity matrix
        sim_vector = self.item_sim_matrix[item_idx].toarray()[0]
        
        # Sort indices by similarity
        # Partition to get top n+1 elements quickly, then sort those
        top_indices = np.argpartition(sim_vector, -(n+1))[-(n+1):]
        top_indices = top_indices[np.argsort(sim_vector[top_indices])[::-1]]
        
        similar_items = []
        for idx in top_indices:
            if idx != item_idx: # Exclude the item itself
                similar_items.append((self.item_inv_mapper[idx], sim_vector[idx]))
                if len(similar_items) == n:
                    break
                    
        return similar_items


def train_svd(ratings_df, model_path="models/svd_model.pkl"):
    """
    Matrix Factorization using Singular Value Decomposition (SVD) via Surprise.
    
    Mathematics:
    SVD decomposes the user-item matrix R into user latent factors (P) and item 
    latent factors (Q). The predicted rating is:
    r_hat_ui = mu + b_u + b_i + dot(q_i, p_u)
    where mu is global mean, and b_u, b_i are user/item biases.
    The model is optimized using Stochastic Gradient Descent (SGD) to minimize 
    regularized squared error.
    """
    print("Preparing data for Surprise SVD...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    print("Splitting into train/test sets...")
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
    
    print("Training SVD model (this may take a while for large datasets)...")
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)
    
    print("Evaluating on test set...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    print("Saving model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, rmse, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Collaborative Filtering models.")
    parser.add_argument("--sample-frac", type=float, default=1.0, 
                        help="Fraction of dataset to use for faster development training")
    args = parser.parse_args()
    
    processed_dir = Path("data/processed")
    ratings_path = processed_dir / "ratings_clean.parquet"
    
    print(f"Loading ratings from {ratings_path}...")
    ratings = pd.read_parquet(ratings_path)
    
    if args.sample_frac < 1.0:
        print(f"Sampling {args.sample_frac * 100}% of the data for faster iteration...")
        ratings = ratings.sample(frac=args.sample_frac, random_state=42)
        
    print(f"Using {len(ratings):,} ratings.")
    
    # 1. Train SVD
    train_svd(ratings, "models/svd_model.pkl")
    
    # UserBased and ItemBased can be fitted on the fly or serialized if needed.
    # For high-performance serving to our Node.js frontend, we serialize the models
    # and similarity matrices to disk for low-latency retrieval. Let's serialize the ItemBasedCF similarities
    # since UserBasedCF sparse matrix can be quite large.
    
    item_cf = ItemBasedCF(min_item_ratings=50)
    item_cf.fit(ratings)
    joblib.dump(item_cf, "models/item_cf_model.pkl")
    print("Saved ItemBasedCF model to models/item_cf_model.pkl")
    
    user_cf = UserBasedCF(k=20, max_ratings_per_user=500)
    user_cf.fit(ratings)
    joblib.dump(user_cf, "models/user_cf_model.pkl")
    print("Saved UserBasedCF model to models/user_cf_model.pkl")
    
    print("Phase 3 complete.")
