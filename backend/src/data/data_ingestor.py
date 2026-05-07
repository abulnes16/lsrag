import os
from datasets import load_dataset

class DataIngestor:
    def __init__(self, retriever):
        self.retriever = retriever
        self.base_data_path = os.getenv("DATA_PATH", "./data")
        
    def ingest_datasets(self, sample_size=50):
        datasets_to_load = [
            ("msmarco-qa", "msmarco_qa_train"),
            ("hotpot-qa", "hotpotqa_train")
        ]
        
        texts_to_insert = []
        
        for folder_name, dataset_name in datasets_to_load:
            cache_dir = os.path.join(self.base_data_path, folder_name)
            try:
                print(f"Loading {dataset_name} from cache at {cache_dir}...")
                dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, cache_dir=cache_dir)
                
                sample_data = dataset['train'].select(range(sample_size)) 
                
                for item in sample_data:
                    # Append string representation. For better RAG performance, we might want to extract specific text fields.
                    texts_to_insert.append(str(item)) 
                    
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                
        if texts_to_insert:
            print(f"Ingesting {len(texts_to_insert)} documents into LightRAG...")
            self.retriever.rag.insert(texts_to_insert)
            print("Ingestion complete.")
        else:
            print("No documents were found to ingest.")
