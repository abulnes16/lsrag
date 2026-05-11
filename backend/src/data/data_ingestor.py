import os
from datasets import load_dataset

class DataIngestor:
    def __init__(self, retriever):
        self.retriever = retriever
        self.base_data_path = os.getenv("DATA_PATH", "./data")
        
    async def ingest_datasets(self, sample_size=50):
        datasets_to_load = [
            ("msmarco-qa", "msmarco-qa"),
            ("hotpot-qa", "hotpotqa")
        ]
        
        texts_to_insert = []
        
        for folder_name, dataset_name in datasets_to_load:
            cache_dir = os.path.join(self.base_data_path, folder_name)
            dataset_path = cache_dir
            try:
                print(f"Loading {dataset_name} from cache at {cache_dir}...")
                dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, cache_dir=cache_dir)
                
                # Determine which split to use
                split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
                total_docs = len(dataset[split_name])
                
                # Calculate sample size: if < 1.0 treat as percentage, else as absolute count
                actual_sample_size = int(total_docs * sample_size) if sample_size < 1.0 else int(sample_size)
                actual_sample_size = max(1, min(actual_sample_size, total_docs))
                
                print(f"Sampling {actual_sample_size} documents ({(actual_sample_size/total_docs)*100:.4f}% of {total_docs}) from {dataset_name}")
                sample_data = dataset[split_name].select(range(actual_sample_size))
                
                for item in sample_data:
                    # In MSMarco and HotpotQA, usually 'question' and 'answers' are relevant
                    # We'll join them into a simple text representation
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
                
        if texts_to_insert:
            print(f"Ingesting {len(texts_to_insert)} documents into LightRAG...")
            await self.retriever.rag.ainsert(texts_to_insert)
            print("Ingestion complete.")
        else:
            print("No documents were found to ingest.")
