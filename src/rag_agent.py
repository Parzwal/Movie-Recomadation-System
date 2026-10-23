import os
import joblib
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# Global cache for FastAPI
_models_cache = None

def get_cached_models():
    global _models_cache
    if _models_cache is not None:
        return _models_cache
        
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import joblib
    
    print("Loading FAISS VectorStore...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.load_local("models/faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("Loading SVD Model...")
    svd = joblib.load("models/svd_model.pkl")
    
    _models_cache = (vs, svd)
    return _models_cache

@tool
def get_personalized_vibe_recommendations(query: str, user_id: int) -> str:
    """
    Finds movies matching a natural language description (e.g. "dark sci-fi with plot twists")
    and personalizes the ranking specifically for the given user's taste.
    """
    vs, svd = get_cached_models()
    
    # 1. Retrieve top 20 semantic matches from FAISS
    docs = vs.similarity_search(query, k=20)
    
    # 2. Score them using SVD for the specific user
    scored_movies = []
    for doc in docs:
        movie_id = doc.metadata['movieId']
        title = doc.metadata['title']
        genres = doc.metadata['genres']
        
        # Predict rating
        pred = svd.predict(user_id, movie_id).est
        scored_movies.append({
            'title': title,
            'genres': genres,
            'predicted_rating': round(pred, 2),
            'vibe': doc.page_content
        })
        
    # 3. Sort by predicted rating descending
    scored_movies.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    # 4. Return top 5 as a formatted string
    top_5 = scored_movies[:5]
    res = f"Top 5 personalized matches for '{query}':\n\n"
    for m in top_5:
        res += f"- {m['title']} (Predicted: {m['predicted_rating']} ⭐)\n  Genres: {m['genres']}\n"
    return res

def create_agent():
    from langgraph.prebuilt import create_react_agent
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment variables.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    
    system_msg_text = (
        "You are an expert movie recommendation AI. You help users find movies they will love. "
        "Always use the 'get_personalized_vibe_recommendations' tool when they ask for movie suggestions based on vibes, plots, or genres. "
        "If they don't provide a user ID, assume User ID 42. "
        "Keep your responses conversational, enthusiastic, and nicely formatted with markdown."
    )
    
    tools = [get_personalized_vibe_recommendations]
    
    # Use modern LangGraph React Agent
    agent = create_react_agent(llm, tools, prompt=system_msg_text)
    return agent
