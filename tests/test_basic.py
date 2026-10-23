import pytest
import pandas as pd
from pathlib import Path
import sys

# Ensure src modules can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import extract_year, clean_title
from src.recommendation_engine import RecommendationEngine

def test_extract_year():
    assert extract_year("Toy Story (1995)") == 1995
    assert extract_year("No Year Mentioned") is pd.isna(pd.NA) or pd.isna(extract_year("No Year Mentioned"))
    
def test_clean_title():
    assert clean_title("Toy Story (1995)") == "Toy Story"
    assert clean_title("Matrix, The (1999)") == "Matrix, The"

def test_recommendation_engine_popular():
    # Only test if models/data are generated
    data_dir = Path("data/processed")
    if (data_dir / "movies_master.parquet").exists():
        engine = RecommendationEngine(data_dir=str(data_dir), models_dir="models")
        recs = engine.top_n_popular(n=5)
        assert len(recs) == 5
        assert 'movieId' in recs[0]
        assert 'score' in recs[0]
