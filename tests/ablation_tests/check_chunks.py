import os
import sys
import time
import asyncio
from datasets import load_dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'backend'))
sys.path.append(os.path.join(project_root, 'backend', 'src'))

from modules.naive_retriever.naive_retriever import NaiveRAGService

def extract_docs(dataset_name, start_idx, end_idx):
    base_data_path = os.path.join(project_root, "data")
    folder_map = {
        "msmarco-qa": "msmarco-qa",
        "hotpotqa": "hotpot-qa"
    }
    folder_name = folder_map[dataset_name]
    cache_dir = os.path.join(base_data_path, folder_name)
    texts_to_insert = []
    
    try:
        print(f"[{dataset_name}] Loading docs from index {start_idx} to {end_idx}...")
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, cache_dir=cache_dir)
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        
        sample_data = dataset[split_name].select(range(start_idx, end_idx))
        
        for item in sample_data:
            text_parts = []
            if 'question' in item: text_parts.append(f"Question: {item['question']}")
            if 'answers' in item: text_parts.append(f"Answers: {item['answers']}")
            if 'context' in item: text_parts.append(f"Context: {item['context']}")
            
            if text_parts:
                texts_to_insert.append("\n".join(text_parts))
            else:
                texts_to_insert.append(str(item)) 
                
        return texts_to_insert
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return []

async def main():
    print("Loading all 5089 documents...")
    # msmarco: 0 to 4580
    # hotpotqa: 0 to 509
    docs_ms = extract_docs("msmarco-qa", 0, 4580)
    docs_hp = extract_docs("hotpotqa", 0, 509)
    all_docs = docs_ms + docs_hp
    print(f"Total documents loaded: {len(all_docs)}")
    
    # Initialize service
    naive_config = {
        "working_dir": os.path.join(project_root, 'data', 'naive_data_temp_test'),
        "llm_model": "phi3:mini",
        "embed_model": "mxbai-embed-large",
        "embed_dim": 1024,
        "chunk_size": 300,
        "chunk_overlap": 50,
    }
    service = NaiveRAGService(naive_config)
    
    print("Chunking documents...")
    all_chunks = []
    for doc in all_docs:
        all_chunks.extend(service.chunk_text(doc))
        
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Test embedding speed on 5 chunks
    print("Testing embedding speed on 5 chunks...")
    start_time = time.time()
    for i in range(min(5, len(all_chunks))):
        await service.client.embed(model=service.embed_model, input=all_chunks[i], truncate=True)
    end_time = time.time()
    avg_time = (end_time - start_time) / min(5, len(all_chunks))
    print(f"Average time per chunk embedding: {avg_time:.4f} seconds")
    estimated_total = (avg_time * len(all_chunks)) / 60
    print(f"Estimated total time to embed {len(all_chunks)} chunks: {estimated_total:.2f} minutes")

if __name__ == "__main__":
    asyncio.run(main())
