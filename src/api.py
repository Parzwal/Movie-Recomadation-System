import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from recommendation_engine import RecommendationEngine

app = FastAPI(
    title="StreamVibe Recommendation API",
    description="High-performance FastAPI backend serving the Node.js Next.js frontend.",
    version="1.0.0"
)

# Enable CORS for the Node.js frontend (default Next.js dev port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine and stats instances
_engine: Optional[RecommendationEngine] = None
_dashboard_stats: Optional[Dict[str, Any]] = None

def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine(
            data_dir=str(PROJECT_DIR / "data" / "processed"),
            models_dir=str(PROJECT_DIR / "models")
        )
    return _engine

def get_dashboard_stats() -> Dict[str, Any]:
    global _dashboard_stats
    if _dashboard_stats is not None:
        return _dashboard_stats
        
    stats_path = PROJECT_DIR / "models" / "dashboard_stats.pkl"
    if stats_path.exists():
        try:
            raw_stats = joblib.load(stats_path)
            top_rated = raw_stats.get('top_rated', pd.DataFrame())
            most_rated = raw_stats.get('most_rated', pd.DataFrame())
            
            _dashboard_stats = {
                'total_movies': int(raw_stats.get('total_movies', 27000)),
                'total_ratings': int(raw_stats.get('total_ratings', 20000000)),
                'avg_rating': float(raw_stats.get('avg_rating', 3.53)),
                'top_rated': top_rated.to_dict(orient='records') if hasattr(top_rated, 'to_dict') else [],
                'most_rated': most_rated.to_dict(orient='records') if hasattr(most_rated, 'to_dict') else []
            }
            return _dashboard_stats
        except Exception as e:
            print(f"Error loading dashboard_stats.pkl: {e}")
            
    _dashboard_stats = {
        'total_movies': 27278,
        'total_ratings': 20000263,
        'avg_rating': 3.53,
        'top_rated': [],
        'most_rated': []
    }
    return _dashboard_stats

class SimilarRequest(BaseModel):
    movie_title: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = 42

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "StreamVibe API",
        "frontend": "Node.js (Next.js)",
        "docs": "/docs"
    }

@app.get("/api/popular")
def get_popular(n: int = Query(15, ge=1, le=100), genre: Optional[str] = None):
    """Returns top popular movies for cold-start / discover carousels."""
    engine = get_engine()
    results = engine.top_n_popular(n=n, genre=genre)
    return {"recommendations": results}

@app.get("/api/dashboard_stats")
def get_stats():
    """Returns precomputed statistics including Top Rated and Most Rated."""
    return get_dashboard_stats()

@app.post("/api/similar")
def find_similar(body: SimilarRequest):
    """Finds movies similar to the given title using Item-Based CF."""
    if not body.movie_title.strip():
        raise HTTPException(status_code=400, detail="Movie title cannot be empty.")
    engine = get_engine()
    res = engine.recommend_similar_movies(body.movie_title.strip(), n=12)
    return res

@app.post("/api/chat")
def chat_with_vibe_ai(body: ChatRequest):
    """AI Assistant endpoint using LangGraph / Gemini or smart fallback."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from rag_agent import create_agent
            agent = create_agent()
            response = agent.invoke({"messages": [("user", body.message)]})
            reply_text = response["messages"][-1].content
            return {"reply": reply_text}
        except Exception as e:
            print(f"Agent error: {e}")
            
    # Intelligent fallback when GEMINI_API_KEY is not set
    engine = get_engine()
    query = body.message.lower()
    
    # Check for genre matches
    genres = ["action", "comedy", "drama", "horror", "romance", "thriller", "sci-fi", "animation"]
    matched_genre = next((g for g in genres if g in query), None)
    
    if matched_genre:
        recs = engine.top_n_popular(n=3, genre=matched_genre.capitalize())
        recs_text = "\n".join([f"- **{m.get('clean_title', m.get('Movie', 'Title'))}** ({m.get('genres', matched_genre.capitalize())})" for m in recs])
        return {
            "reply": f"Based on your vibe for **{matched_genre}**, here are top recommendations:\n\n{recs_text}\n\nEnjoy watching!"
        }
    
    # Generic recommendations fallback
    recs = engine.top_n_popular(n=3)
    recs_text = "\n".join([f"- **{m.get('clean_title', m.get('Movie', 'Title'))}** ({m.get('score', 4.5)} ⭐)" for m in recs])
    return {
        "reply": f"I'd love to help you find the perfect movie! Here are some critically acclaimed favorites:\n\n{recs_text}\n\nTell me more about the plot or genre you're craving!"
    }

@app.get("/api/poster")
def get_movie_poster(title: str, imdb_id: Optional[str] = None):
    """Generates an aesthetic SVG movie poster placeholder matching the StreamVibe UI."""
    clean = title.strip()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" viewBox="0 0 300 450">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1f1f23" />
      <stop offset="50%" stop-color="#141414" />
      <stop offset="100%" stop-color="#0d0d0f" />
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#E50914" />
      <stop offset="100%" stop-color="#B20710" />
    </linearGradient>
  </defs>
  <rect width="300" height="450" fill="url(#grad)" rx="8" />
  <rect x="0" y="442" width="300" height="8" fill="url(#accent)" />
  <circle cx="150" cy="180" r="48" fill="#E50914" fill-opacity="0.12" stroke="#E50914" stroke-width="2" stroke-dasharray="4 4"/>
  <text x="150" y="192" font-family="system-ui, -apple-system, sans-serif" font-size="38" text-anchor="middle" fill="#E50914">🎬</text>
  <text x="150" y="270" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#FFFFFF">
    {clean[:24]}
  </text>
  <text x="150" y="295" font-family="system-ui, -apple-system, sans-serif" font-size="12" text-anchor="middle" fill="#888888">
    STREAMVIBE ORIGINAL
  </text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
