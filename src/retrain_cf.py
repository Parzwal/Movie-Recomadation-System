import pandas as pd
import joblib
from pathlib import Path
from surprise import Dataset, Reader, SVD
from collaborative_filtering import ItemBasedCF, UserBasedCF

def retrain_models():
    print("Loading ratings data...")
    ratings = pd.read_parquet("data/processed/ratings_clean.parquet")
    
    print("Fitting ItemBasedCF...")
    item_cf = ItemBasedCF(min_item_ratings=50)
    item_cf.fit(ratings)
    joblib.dump(item_cf, "models/item_cf_model.pkl")
    
    print("Fitting UserBasedCF...")
    user_cf = UserBasedCF(k=20, max_ratings_per_user=500)
    user_cf.fit(ratings)
    joblib.dump(user_cf, "models/user_cf_model.pkl")
    
    print("Fitting SVD Model (Matrix Factorization)...")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.04, random_state=42)
    svd.fit(trainset)
    joblib.dump(svd, "models/svd_model.pkl")
    
    print("Done. All models saved to models/ directory.")

if __name__ == "__main__":
    retrain_models()
