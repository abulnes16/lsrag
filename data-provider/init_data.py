import os
import sys
from datasets import load_dataset

def download_datasets():
    print("Downloading FlashRAG datasets: MSMarco-QA and Hotpot-QA")
    # This is an example initialization script.
    # FlashRAG typically manages its own download methods or uses HuggingFace datasets.
    # Here, specific calls can be added to initialize the shared folder '/app/data'
    
    data_dir = "/app/data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Example using the HuggingFace datasets library if they are uploaded there
        # from datasets import load_dataset
        load_dataset(
            "RUC-NLPIR/FlashRAG_datasets", 
            "msmarco-qa", 
            cache_dir=os.path.join(data_dir, "msmarco-qa")
        )
        load_dataset(
            "RUC-NLPIR/FlashRAG_datasets",
            "hotpotqa", 
            cache_dir=os.path.join(data_dir, "hotpot-qa")
        )
        print(f"Data will be stored in: {data_dir}")
        print("The download of MSMarco-QA and Hotpot-QA has been orchestrated.")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_datasets()
