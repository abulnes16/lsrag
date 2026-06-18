import os
import sys
import time
import asyncio
import json
import xml.etree.ElementTree as ET

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(os.path.join(project_root, 'backend'))
sys.path.append(os.path.join(project_root, 'backend', 'src'))

from datasets import load_dataset
from src.modules.light_retriever.light_retriever import LightRAGService

# Batch 4 took 9 minutes and has already run; add this to the current time
def get_graph_size(test_dir):
    """Parses the graphml file to count nodes and relationships quickly."""
    graphml_path = os.path.join(test_dir, "graph_chunk_entity_relation.graphml")
    if not os.path.exists(graphml_path):
        return 0, 0
    try:
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        ns = {'gml': 'http://graphml.graphdrawing.org/xmlns'}
        nodes = len(root.findall('.//gml:node', ns))
        edges = len(root.findall('.//gml:edge', ns))
        return nodes, edges
    except Exception as e:
        print(f"Error parsing graph: {e}")
        return 0, 0

def extract_docs(dataset_name, start_idx, end_idx):
    """Extract documents for incremental testing"""
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

async def run_incremental_tests():
    print("=== Starting Index Incremental Tests ===")
    
    test_dir = os.path.join(project_root, "data", "lightrag_cache")
    if not os.path.exists(test_dir):
        print(f"WARNING: We didn't found the test directory {test_dir}. Check the route.")
        
    print(f"Initializing LightRAG in {test_dir} (Index of 889 documents)...")
    
    config = {
        "working_dir": test_dir,
        "llm_model": "llama3.2:3b",
        "embed_model": "mxbai-embed-large",
        "embed_dim": 1024,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "timeout": 1200,
        "max_async": 1,
    }
    
    light_retriever = LightRAGService(config)

    await light_retriever.rag.initialize_storages()
    
    # Generate batches with increasing sizes: 100, 200, 300, 400, 500, 600, 500
    # to add exactly 2600 new docs and reach 5089 total documents (2489 + 2600 = 5089).
    sizes = [100, 200, 300, 400, 500, 600, 500]
    batches = []
    
    current_ms_idx = 2240
    current_hp_idx = 249
    
    for i, size in enumerate(sizes):
        batch_num = 17 + i
        ms_size = int(size * 0.9)
        hp_size = int(size * 0.1)
        
        ms_start = current_ms_idx
        ms_end = current_ms_idx + ms_size
        hp_start = current_hp_idx
        hp_end = current_hp_idx + hp_size
        
        batches.append({
            "batch_num": batch_num,
            "msmarco": (ms_start, ms_end),
            "hotpotqa": (hp_start, hp_end)
        })
        
        current_ms_idx = ms_end
        current_hp_idx = hp_end
    
    json_output_path = os.path.join(os.path.dirname(__file__), "incremental_tests_results_2.json")
    results_log = []
    
    # Load previous results. If it does not exist, read from the original incremental_test_results.json file
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                results_log = json.load(f)
            print(f"Loaded {len(results_log)} previous results from the new JSON file: {json_output_path}")
        except Exception as e:
            print(f"Error loading existing JSON: {e}")
    else:
        source_path = os.path.join(os.path.dirname(__file__), "incremental_test_results.json")
        if os.path.exists(source_path):
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    results_log = json.load(f)
                print(f"Initialized results from the original file: {source_path}")
            except Exception as e:
                print(f"Error loading the original results file: {e}")

    # Calculate current total documents from the loaded history
    current_total_docs = 2489
    if results_log:
        # Filter if there are elements with Total_Docs_In_Graph
        current_total_docs = results_log[-1]["Total_Docs_In_Graph"]

    for batch in batches:
        print(f"\n--- Preparing Batch {batch['batch_num']} ---")
        
        docs_msmarco = extract_docs("msmarco-qa", batch["msmarco"][0], batch["msmarco"][1])
        docs_hotpotqa = extract_docs("hotpotqa", batch["hotpotqa"][0], batch["hotpotqa"][1])
        
        batch_texts = docs_msmarco + docs_hotpotqa
        
        print(f"Batch {batch['batch_num']} ready with {len(batch_texts)} documents.")
        print("Starting indexing...")
        
        start_time = time.time()
        
        await light_retriever.rag.ainsert(batch_texts)
        
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        
        # Extract updated graph size
        nodes_count, edges_count = get_graph_size(test_dir)
        
        current_total_docs += len(batch_texts)
        
        print(f"Batch {batch['batch_num']} completed in {elapsed_minutes:.2f} minutes. Graph size: {nodes_count} nodes, {edges_count} edges.")
        
        results_log.append({
            "Batch_Num": batch['batch_num'],
            "New_Docs": len(batch_texts),
            "Total_Docs_In_Graph": current_total_docs,
            "Time_Minutes": elapsed_minutes,
            "Graph_Nodes": nodes_count,
            "Graph_Edges": edges_count
        })

    print("\n=== FINAL SUMMARY OF THE SCALABILITY TEST ===")
    print("Batch_Num | New_Docs | Total_Docs_In_Graph | Time_Minutes | Nodes | Edges")
    print("-" * 75)
    for res in results_log:
        print(f"{res['Batch_Num']:4d} | {res['New_Docs']:11d} | {res['Total_Docs_In_Graph']:21d} | {res['Time_Minutes']:12.2f} | {res.get('Graph_Nodes', 0) or 0:5d} | {res.get('Graph_Edges', 0) or 0:5d}")
    
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=4, ensure_ascii=False)
        print(f"\n[+] Results exported successfully in JSON format: {json_output_path}")
    except Exception as e:
        print(f"\n[-] Error exporting results in JSON format: {e}")

    print("\nIncremental test completed! Your main index now has 5089 documents ready for ablation tests.")

if __name__ == "__main__":
    asyncio.run(run_incremental_tests())
