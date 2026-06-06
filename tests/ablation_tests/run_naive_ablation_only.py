import sys
import os
import json
import time
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from the script's directory (contains OPENAI_API_KEY)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Set default OLLAMA_HOST to localhost for local terminal execution (outside docker)
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

# Add backend to python path so we can import LightRAG and RAGMetrics
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'backend', 'src'))
sys.path.append(os.path.join(project_root, 'backend'))

from modules.naive_retriever.naive_retriever import NaiveRAGService
from modules.metrics.rag_metrics import RAGMetrics

VARIANTS = ["naive", "local", "global", "mix"]

async def run_naive_update_for_file(filepath: str, is_corporate: bool):
    print(f"\n--- Starting NaiveRAG update for {os.path.basename(filepath)} ---")
    
    output_final = filepath.replace(".json", "_results_final.json")
    output_intermediate = filepath.replace(".json", "_results_intermediate.json")
    
    results = []
    
    # Try to load existing data
    if os.path.exists(output_final):
        with open(output_final, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("details", [])
        print(f"Loaded {len(results)} existing questions from final output: {output_final}")
    elif os.path.exists(output_intermediate):
        with open(output_intermediate, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing questions from intermediate output: {output_intermediate}")
    else:
        # If no file exists, we read the base questions list
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        results = [{
            "query": q["query"],
            "reference": q["reference"],
            "category": q.get("category", "unknown"),
            "variants": {}
        } for q in questions]
        print(f"No existing results found. Created fresh details structure for {len(results)} questions.")

    # Initialize NaiveRAG Service
    naive_config = {
        "working_dir": os.path.join(project_root, 'data', "corporate_naive_data" if is_corporate else "naive_data"),
        "llm_model": "phi3:mini",
        "embed_model": "mxbai-embed-large",
        "embed_dim": 1024,
        "chunk_size": 300,
        "chunk_overlap": 50,
    }
    print(f"Initializing NaiveRAG Service using working_dir: {naive_config['working_dir']}...")
    naive_service = NaiveRAGService(naive_config)
    
    print("Initializing RAGMetrics Service (requires OPENAI_API_KEY for GPT-4o-mini)...")
    metrics_service = RAGMetrics()
    
    for i, q_results in enumerate(results):
        query = q_results["query"]
        reference = q_results["reference"]
        category = q_results.get("category", "unknown")
        
        print(f"\n[{i+1}/{len(results)}] Query: {query}")
        print("  -> Running NaiveRAG variant...")
        
        try:
            start_time = time.time()
            search_results = await naive_service.search(query, num=5)
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
            
            q_results["variants"]["naive"] = {
                "answer": answer,
                "response_time_seconds": response_time,
                "metrics": metrics
            }
        except Exception as e:
            print(f"  -> Error in NaiveRAG: {e}")
            q_results["variants"]["naive"] = {
                "answer": f"ERROR: {str(e)}",
                "response_time_seconds": 0.0,
                "metrics": {"faithfulness": 0.0, "recall": 0.0, "relevancy": 0.0, "ndcg": 0.0}
            }
            
        # Save intermediate
        with open(output_intermediate, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    # Recalculate Averages
    print("\n--- Recalculating Averages ---")
    averages = {
        variant: {
            "response_time_seconds": 0.0,
            "faithfulness": 0.0,
            "recall": 0.0,
            "relevancy": 0.0,
            "ndcg": 0.0
        } for variant in VARIANTS
    }
    
    # Get all categories
    categories = list(set([r["category"] for r in results]))
    category_averages = {
        cat: {
            variant: {
                "response_time_seconds": 0.0,
                "faithfulness": 0.0,
                "recall": 0.0,
                "relevancy": 0.0,
                "ndcg": 0.0
            } for variant in VARIANTS
        } for cat in categories
    }
    category_counts = {cat: 0 for cat in categories}
    
    for r in results:
        cat = r["category"]
        category_counts[cat] += 1
        for variant in VARIANTS:
            v_data = r["variants"].get(variant, {})
            
            resp_time = v_data.get("response_time_seconds", 0.0)
            mets = v_data.get("metrics", {})
            f_score = mets.get("faithfulness") or 0.0
            r_score = mets.get("recall") or 0.0
            rel_score = mets.get("relevancy") or 0.0
            n_score = mets.get("ndcg") or 0.0
            
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
        
    print(f"\n[+] Updated results saved to {output_final}")

async def main():
    print("=====================================================")
    print(" N A I V E R A G   -   O N L Y   R E - E V A L U A T I O N")
    print("=====================================================\n")
    print("WARNING: This script will ONLY re-run queries and calculate metrics for NaiveRAG.")
    print("Existing results for LSRAG variants (local, global, mix) will be preserved.\n")
    
    base_dir = os.path.dirname(__file__)
    
    general_file = os.path.join(base_dir, "evaluation_questions.json")
    corp_file = os.path.join(base_dir, "evaluation_questions_corporate.json")
    
    if os.path.exists(general_file):
        await run_naive_update_for_file(general_file, is_corporate=False)
    else:
        print(f"File not found: {general_file}")
        
    if os.path.exists(corp_file):
        await run_naive_update_for_file(corp_file, is_corporate=True)
    else:
        print(f"File not found: {corp_file}")

if __name__ == "__main__":
    asyncio.run(main())
