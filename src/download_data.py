import os
import zipfile
import urllib.request
from pathlib import Path

def download_dataset(dataset_name="ml-20m"):
    """
    Downloads and extracts the MovieLens dataset.
    Falls back to ml-latest-small if ml-20m fails.
    """
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url_20m = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    url_small = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    target_url = url_20m if dataset_name == "ml-20m" else url_small
    zip_path = data_dir / f"{dataset_name}.zip"
    extract_path = data_dir
    
    print(f"Attempting to download {dataset_name} from {target_url}...")
    try:
        urllib.request.urlretrieve(target_url, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted to {extract_path}/{dataset_name}")
        
        # Clean up zip file
        os.remove(zip_path)
        print("Dataset ready.")
        
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")
        if dataset_name == "ml-20m":
            print("Falling back to ml-latest-small...")
            download_dataset("ml-latest-small")

if __name__ == "__main__":
    download_dataset()
