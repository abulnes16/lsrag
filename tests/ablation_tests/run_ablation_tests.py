import sys
import os
import json
import time
import asyncio
from typing import List, Dict

# Add backend to python path so we can import LightRAG and RAGMetrics
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'backend', 'src'))
sys.path.append(os.path.join(project_root, 'backend'))

from modules.light_retriever.light_retriever import LightRAGService
from modules.naive_retriever.naive_retriever import NaiveRAGService
from modules.metrics.rag_metrics import RAGMetrics

VARIANTS = ["naive", "local", "global", "mix"]

async def run_evaluation_for_file(filepath: str, is_corporate: bool):
    print(f"\n--- Starting evaluation for {os.path.basename(filepath)} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        questions = json.load(f)
        
    cache_dir = "corporate_lightrag_cache" if is_corporate else "lightrag_cache"
    config = {
        "working_dir": os.path.join(project_root, 'data', cache_dir),
        # You can adjust these models if your instance uses phi3:mini or another one
        "llm_model": "llama3.2:3b", 
        "embed_model": "mxbai-embed-large",
        "chunk_size": 1200,
        "chunk_overlap": 100
    }
    
    print("Initializing LightRAG Service...")
    rag_service = LightRAGService(config)
    print("Initializing NaiveRAG Service...")
    
    naive_config = {
        "working_dir": os.path.join(project_root, 'data', "corporate_naive_data" if is_corporate else "naive_data"),
        "llm_model": config["llm_model"],
        "embed_model": config["embed_model"],
        "embed_dim": 1024,
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"]
    }
    naive_service = NaiveRAGService(naive_config)
    
    print("Initializing RAGMetrics Service (requires OPENAI_API_KEY for GPT-4o-mini)...")
    metrics_service = RAGMetrics()
    
    results = []
    output_intermediate = filepath.replace(".json", "_results_intermediate.json")
    output_final = filepath.replace(".json", "_results_final.json")
    
    # Check if we have progress
    if os.path.exists(output_intermediate):
        with open(output_intermediate, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} previously evaluated questions.")
        
    start_index = len(results)
    
    for i in range(start_index, len(questions)):
        q_data = questions[i]
        query = q_data["query"]
        reference = q_data["reference"]
        category = q_data.get("category", "unknown")
        
        print(f"\n[{i+1}/{len(questions)}] Query: {query}")
        
        q_results = {
            "query": query,
            "reference": reference,
            "category": category,
            "variants": {}
        }
        
        for variant in VARIANTS:
            print(f"  -> Running Variant: {variant}")
            try:
                start_time = time.time()
                # Run search
                if variant == "naive":
                    search_results = await naive_service.search(query, num=5)
                else:
                    search_results = await rag_service.search(query, num=5, mode=variant)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Extract answer and context
                if isinstance(search_results, list) and len(search_results) > 0:
                    answer = search_results[0].get("contents", "")
                    contexts = search_results[0].get("contexts", [])
                else:
                    answer = str(search_results)
                    contexts = []
                
                print("     Calculating metrics...")
                metrics = await metrics_service.calculate_all_metrics(
                    question=query, 
                    answer=answer, 
                    contexts=contexts, 
                    reference=reference
                )
                
                q_results["variants"][variant] = {
                    "answer": answer,
                    "response_time_seconds": response_time,
                    "metrics": metrics
                }
            except Exception as e:
                print(f"  -> Error in Variant {variant}: {e}")
                q_results["variants"][variant] = {
                    "answer": f"ERROR: {str(e)}",
                    "response_time_seconds": 0,
                    "metrics": {"faithfulness": 0, "recall": 0, "relevancy": 0, "ndcg": 0}
                }
                
        results.append(q_results)
        
        # Save intermediate
        with open(output_intermediate, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    # Calculate Averages
    print("\n--- Calculating Averages ---")
    averages = {
        variant: {
            "response_time_seconds": 0,
            "faithfulness": 0,
            "recall": 0,
            "relevancy": 0,
            "ndcg": 0
        } for variant in VARIANTS
    }
    
    # Also calculate averages by category
    categories = list(set([r["category"] for r in results]))
    category_averages = {
        cat: {
            variant: {
                "response_time_seconds": 0,
                "faithfulness": 0,
                "recall": 0,
                "relevancy": 0,
                "ndcg": 0
            } for variant in VARIANTS
        } for cat in categories
    }
    category_counts = {cat: 0 for cat in categories}
    
    for r in results:
        cat = r["category"]
        category_counts[cat] += 1
        for variant in VARIANTS:
            v_data = r["variants"].get(variant, {})
            
            resp_time = v_data.get("response_time_seconds", 0)
            mets = v_data.get("metrics", {})
            f_score = mets.get("faithfulness") or 0
            r_score = mets.get("recall") or 0
            rel_score = mets.get("relevancy") or 0
            n_score = mets.get("ndcg") or 0
            
            # Global avgs
            averages[variant]["response_time_seconds"] += resp_time
            averages[variant]["faithfulness"] += f_score
            averages[variant]["recall"] += r_score
            averages[variant]["relevancy"] += rel_score
            averages[variant]["ndcg"] += n_score
            
            # Category avgs
            category_averages[cat][variant]["response_time_seconds"] += resp_time
            category_averages[cat][variant]["faithfulness"] += f_score
            category_averages[cat][variant]["recall"] += r_score
            category_averages[cat][variant]["relevancy"] += rel_score
            category_averages[cat][variant]["ndcg"] += n_score
            
    num_questions = len(results)
    if num_questions > 0:
        for variant in VARIANTS:
            for k in averages[variant]:
                averages[variant][k] /= num_questions
                
        for cat in categories:
            count = category_counts[cat]
            if count > 0:
                for variant in VARIANTS:
                    for k in category_averages[cat][variant]:
                        category_averages[cat][variant][k] /= count
                
    final_output = {
        "global_averages": averages,
        "category_averages": category_averages,
        "details": results
    }
    
    with open(output_final, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\nFinal results saved to {output_final}")

async def main():
    print("=========================================")
    print(" L S R A G  -  A B L A T I O N   T E S T S")
    print("=========================================\n")
    print("WARNING: This script requires OPENAI_API_KEY environment variable to be set.")
    print("WARNING: This process will take several hours. Progress is saved automatically.\n")
    
    base_dir = os.path.dirname(__file__)
    
    general_file = os.path.join(base_dir, "evaluation_questions.json")
    corp_file = os.path.join(base_dir, "evaluation_questions_corporate.json")
    
    if os.path.exists(general_file):
        await run_evaluation_for_file(general_file, is_corporate=False)
    else:
        print(f"File not found: {general_file}")
        
    if os.path.exists(corp_file):
        await run_evaluation_for_file(corp_file, is_corporate=True)
    else:
        print(f"File not found: {corp_file}")

if __name__ == "__main__":
    asyncio.run(main())
