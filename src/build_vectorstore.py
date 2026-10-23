import pandas as pd
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import time

def build_vectorstore(raw_dir="data/raw/ml-20m", processed_dir="data/processed", out_dir="models"):
    print("Loading movies...")
    movies = pd.read_parquet(Path(processed_dir) / "movies_master.parquet")
    
    print("Loading genome tags...")
    genome_tags = pd.read_csv(Path(raw_dir) / "genome-tags.csv")
    genome_scores = pd.read_csv(Path(raw_dir) / "genome-scores.csv")
    
    # Filter for high relevance tags (e.g. score > 0.8)
    print("Filtering genome scores (relevance > 0.8)...")
    high_scores = genome_scores[genome_scores['relevance'] > 0.8]
    
    # Merge with tag names
    merged_tags = high_scores.merge(genome_tags, on='tagId')
    
    # Group tags by movieId
    print("Grouping tags by movie...")
    movie_tags = merged_tags.groupby('movieId')['tag'].apply(lambda x: ', '.join(x)).reset_index()
    
    # Merge with movies
    movies_with_tags = movies.merge(movie_tags, on='movieId', how='left')
    movies_with_tags['tag'] = movies_with_tags['tag'].fillna('No specific tags')
    
    print("Building Document objects...")
    documents = []
    for _, row in movies_with_tags.iterrows():
        # Vibe string
        text = f"Title: {row['clean_title']}. Genres: {row['genres']}. Description/Vibes: {row['tag']}."
        doc = Document(
            page_content=text,
            metadata={
                'movieId': int(row['movieId']),
                'title': row['clean_title'],
                'genres': row['genres']
            }
        )
        documents.append(doc)
        
    print(f"Created {len(documents)} documents. Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS VectorStore... (This may take 1-3 minutes)")
    start_time = time.time()
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"FAISS index built in {time.time() - start_time:.1f} seconds.")
    
    out_path = Path(out_dir) / "faiss_index"
    print(f"Saving to {out_path}...")
    vectorstore.save_local(str(out_path))
    print("Done!")

if __name__ == "__main__":
    build_vectorstore()
