import os
from datasets import load_dataset

class DataIngestor:
    def __init__(self, light_retriever=None, naive_retriever=None):
        self.light_retriever = light_retriever
        self.naive_retriever = naive_retriever
        self.base_data_path = os.getenv("DATA_PATH", "./data")
        
    def _get_texts(self, sample_size=50):
        datasets_to_load = [
            ("msmarco-qa", "msmarco-qa"),
            ("hotpot-qa", "hotpotqa")
        ]
        
        texts_to_insert = []
        
        for folder_name, dataset_name in datasets_to_load:
            cache_dir = os.path.join(self.base_data_path, folder_name)
            try:
                print(f"Loading {dataset_name} from cache at {cache_dir}...")
                dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, cache_dir=cache_dir)
                
                split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
                total_docs = len(dataset[split_name])
                
                actual_sample_size = int(total_docs * sample_size) if sample_size < 1.0 else int(sample_size)
                actual_sample_size = max(1, min(actual_sample_size, total_docs))
                
                print(f"Sampling {actual_sample_size} documents ({(actual_sample_size/total_docs)*100:.4f}% of {total_docs}) from {dataset_name}")
                sample_data = dataset[split_name].select(range(actual_sample_size))
                
                for item in sample_data:
                    text_parts = []
                    if 'question' in item: text_parts.append(f"Question: {item['question']}")
                    if 'answers' in item: text_parts.append(f"Answers: {item['answers']}")
                    if 'context' in item: text_parts.append(f"Context: {item['context']}")
                    
                    if text_parts:
                        texts_to_insert.append("\n".join(text_parts))
                    else:
                        texts_to_insert.append(str(item)) 
                    
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                
        return texts_to_insert

    def _get_corporate_texts(self):
        corporate_dir = os.getenv("CORPORATE_DATASET_PATH", "/app/corporate_dataset")
        texts_to_insert = []
        if not os.path.exists(corporate_dir):
            print(f"Corporate dataset directory not found at: {corporate_dir}")
            return texts_to_insert
            
        print(f"Loading corporate dataset from {corporate_dir}...")
        for root, dirs, files in os.walk(corporate_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                texts_to_insert.append(content)
                                print(f"Loaded corporate file: {file}")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
        return texts_to_insert

    async def ingest_datasets(self, sample_size=50):
        """Maintains backwards compatibility, runs both if configured."""
        if os.getenv("LIGHTRAG_DIR", "lightrag_cache") == "corporate_lightrag_cache":
            texts = self._get_corporate_texts()
        else:
            texts = self._get_texts(sample_size)
        
        if not texts:
            print("No documents were found to ingest.")
            return

        if self.light_retriever:
            print(f"\n--- Ingesting {len(texts)} documents into LightRAG ---")
            await self.light_retriever.rag.ainsert(texts)
            print("LightRAG Ingestion complete.")
            
        if self.naive_retriever:
            print(f"\n--- Ingesting {len(texts)} documents into NaiveRAG ---")
            await self.naive_retriever.initialize(texts)
            print("NaiveRAG Ingestion complete.")
