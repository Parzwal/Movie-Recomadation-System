import pandas as pd
import numpy as np
import time
from pathlib import Path
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import train_test_split
from collaborative_filtering import UserBasedCF, ItemBasedCF
import joblib

def precision_recall_at_k_custom(model, test_data, user_ids, k=10, threshold=4.0, model_type='item'):
    """
    Evaluates P@K and R@K for custom models on a subset of users.
    """
    precisions = []
    recalls = []
    
    # Pre-group test items for fast lookup
    test_user_groups = test_data.groupby('userId')
    
    for uid in user_ids:
        if uid not in test_user_groups.groups:
            continue
            
        true_items = test_user_groups.get_group(uid)
        relevant_items = set(true_items[true_items['rating'] >= threshold]['movieId'])
        
        n_rel = len(relevant_items)
        if n_rel == 0:
            continue
            
        # Get top K recommendations
        if model_type == 'item':
            # For item-based, we get items similar to what the user liked in train data
            # Here we just mock the ranking logic based on item similarity for evaluation
            # It's faster to evaluate SVD and Hybrid natively
            pass # We'll handle this in a unified predict loop
    
    return 0, 0

def unified_precision_recall_at_k(train_df, test_df, svd_model, item_cf, user_cf, k=10, threshold=4.0, n_users=1000):
    """
    Evaluates all 4 methods (SVD, Item-CF, User-CF, Hybrid) on a common subset of users
    for a fair apples-to-apples P@K and R@K comparison.
    """
    # Sample test users who have at least some relevant items
    test_relevant = test_df[test_df['rating'] >= threshold]
    valid_users = test_relevant['userId'].unique()
    
    np.random.seed(42)
    sample_users = np.random.choice(valid_users, min(n_users, len(valid_users)), replace=False)
    
    test_user_groups = test_df.groupby('userId')
    train_user_groups = train_df.groupby('userId')
    
    results = {
        'SVD': {'p': [], 'r': []},
        'Item-CF': {'p': [], 'r': []},
        'User-CF': {'p': [], 'r': []},
        'Hybrid': {'p': [], 'r': []}
    }
    
    all_items = set(train_df['movieId'].unique())
    
    print(f"Evaluating P@K and R@K on {len(sample_users)} users...")
    
    for idx, uid in enumerate(sample_users):
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{len(sample_users)} users...")
            
        true_ratings = test_user_groups.get_group(uid)
        relevant_items = set(true_ratings[true_ratings['rating'] >= threshold]['movieId'])
        
        n_rel = len(relevant_items)
        if n_rel == 0:
            continue
            
        train_items = set()
        if uid in train_user_groups.groups:
            train_items = set(train_user_groups.get_group(uid)['movieId'])
            
        candidate_items = list(all_items - train_items)
        
        # 1. SVD Predictions
        svd_preds = []
        for iid in candidate_items:
            svd_preds.append((iid, svd_model.predict(uid, iid).est))
        svd_preds.sort(key=lambda x: x[1], reverse=True)
        svd_top_k = set([x[0] for x in svd_preds[:k]])
        
        # 2. Item-CF Predictions
        item_scores = defaultdict(float)
        # Aggregate similarity from user's highly rated train items
        if uid in train_user_groups.groups:
            user_train = train_user_groups.get_group(uid)
            liked_train = user_train[user_train['rating'] >= threshold]['movieId']
            for liked_item in liked_train:
                sims = item_cf.get_similar_items(liked_item, n=20)
                for sim_item, score in sims:
                    if sim_item not in train_items:
                        item_scores[sim_item] += score
        item_preds = list(item_scores.items())
        item_preds.sort(key=lambda x: x[1], reverse=True)
        item_top_k = set([x[0] for x in item_preds[:k]])
        
        # 3. User-CF Predictions
        user_scores = defaultdict(float)
        sim_users = user_cf.get_similar_users(uid, n=20)
        for sim_uid, sim_score in sim_users:
            if sim_uid in train_user_groups.groups:
                sim_user_train = train_user_groups.get_group(sim_uid)
                for _, row in sim_user_train.iterrows():
                    if row['movieId'] not in train_items:
                        user_scores[row['movieId']] += sim_score * row['rating']
        user_preds = list(user_scores.items())
        user_preds.sort(key=lambda x: x[1], reverse=True)
        user_top_k = set([x[0] for x in user_preds[:k]])
        
        # 4. Hybrid (SVD + Item-CF)
        # Normalize scores to combine them
        hybrid_scores = {}
        if svd_preds:
            max_svd = max([x[1] for x in svd_preds]) or 1.0
        if item_preds:
            max_item = max([x[1] for x in item_preds]) or 1.0
            
        svd_dict = dict(svd_preds)
        item_dict = dict(item_preds)
        
        for iid in candidate_items:
            s_score = svd_dict.get(iid, 0) / (max_svd if 'max_svd' in locals() else 1)
            i_score = item_dict.get(iid, 0) / (max_item if 'max_item' in locals() else 1)
            hybrid_scores[iid] = (0.7 * s_score) + (0.3 * i_score)
            
        hybrid_preds = list(hybrid_scores.items())
        hybrid_preds.sort(key=lambda x: x[1], reverse=True)
        hybrid_top_k = set([x[0] for x in hybrid_preds[:k]])
        
        # Compute metrics
        for model_name, top_k_set in [('SVD', svd_top_k), ('Item-CF', item_top_k), ('User-CF', user_top_k), ('Hybrid', hybrid_top_k)]:
            n_rel_and_rec = len(top_k_set & relevant_items)
            results[model_name]['p'].append(n_rel_and_rec / k)
            results[model_name]['r'].append(n_rel_and_rec / n_rel)

    final_metrics = {}
    for model_name in results:
        p = np.mean(results[model_name]['p']) if results[model_name]['p'] else 0
        r = np.mean(results[model_name]['r']) if results[model_name]['r'] else 0
        final_metrics[model_name] = {'Precision@10': p, 'Recall@10': r}
        
    return final_metrics
    
from collections import defaultdict

def evaluate_full_dataset():
    print("Loading full dataset...")
    ratings = pd.read_parquet("data/processed/ratings_clean.parquet")
    
    print("Splitting train/test (80/20)...")
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    
    metrics = []
    
    # 1. SVD
    print("\n--- Training SVD (Full 20M) ---")
    start_time = time.time()
    reader = Reader(rating_scale=(0.5, 5.0))
    data_train = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    trainset = data_train.build_full_trainset()
    
    svd = SVD(n_factors=50, n_epochs=25, lr_all=0.005, reg_all=0.04, random_state=42)
    svd.fit(trainset)
    svd_time = time.time() - start_time
    
    # SVD RMSE/MAE
    print("Evaluating SVD RMSE/MAE...")
    testset_tuples = [tuple(x) for x in test_df[['userId', 'movieId', 'rating']].values]
    svd_preds = svd.test(testset_tuples)
    from surprise import accuracy
    rmse_svd = accuracy.rmse(svd_preds, verbose=False)
    mae_svd = accuracy.mae(svd_preds, verbose=False)
    
    # 2. Item-CF
    print("\n--- Training Item-based CF ---")
    start_time = time.time()
    item_cf = ItemBasedCF(min_item_ratings=50)
    item_cf.fit(train_df)
    item_time = time.time() - start_time
    
    # 3. User-CF
    print("\n--- Training User-based CF ---")
    start_time = time.time()
    user_cf = UserBasedCF(k=20, max_ratings_per_user=500)
    user_cf.fit(train_df)
    user_time = time.time() - start_time
    
    # Compute P@K and R@K using 1000 users
    print("\n--- Computing Ranking Metrics (P@10, R@10) ---")
    ranking_metrics = unified_precision_recall_at_k(train_df, test_df, svd, item_cf, user_cf, k=10, threshold=4.0, n_users=1000)
    
    # Compile Results
    metrics.append({
        'Model': 'SVD (Tuned)',
        'RMSE': rmse_svd,
        'MAE': mae_svd,
        'Precision@10': ranking_metrics['SVD']['Precision@10'],
        'Recall@10': ranking_metrics['SVD']['Recall@10'],
        'Train Time (s)': svd_time
    })
    
    metrics.append({
        'Model': 'Item-based CF',
        'RMSE': np.nan, # Custom ranking models don't predict ratings
        'MAE': np.nan,
        'Precision@10': ranking_metrics['Item-CF']['Precision@10'],
        'Recall@10': ranking_metrics['Item-CF']['Recall@10'],
        'Train Time (s)': item_time
    })
    
    metrics.append({
        'Model': 'User-based CF',
        'RMSE': np.nan,
        'MAE': np.nan,
        'Precision@10': ranking_metrics['User-CF']['Precision@10'],
        'Recall@10': ranking_metrics['User-CF']['Recall@10'],
        'Train Time (s)': user_time
    })
    
    metrics.append({
        'Model': 'Hybrid (SVD + Item-CF)',
        'RMSE': np.nan,
        'MAE': np.nan,
        'Precision@10': ranking_metrics['Hybrid']['Precision@10'],
        'Recall@10': ranking_metrics['Hybrid']['Recall@10'],
        'Train Time (s)': svd_time + item_time
    })
    
    df_results = pd.DataFrame(metrics)
    for col in ['RMSE', 'MAE', 'Precision@10', 'Recall@10']:
        df_results[col] = df_results[col].round(4)
    df_results['Train Time (s)'] = df_results['Train Time (s)'].round(1)
    
    print("\n--- Final Evaluation Results (Full 20M) ---")
    print(df_results.to_markdown(index=False))
    
    out_dir = Path("data/processed")
    df_results.to_csv(out_dir / "evaluation_metrics_20m.csv", index=False)
    with open(out_dir / "evaluation_metrics_20m.md", "w") as f:
        f.write("# Final Model Evaluation Results (Full 20M)\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n*Evaluation performed on the full MovieLens 20M dataset. Ranking metrics (Precision/Recall@10) computed on a random sample of 1,000 test users.*")

if __name__ == "__main__":
    evaluate_full_dataset()
