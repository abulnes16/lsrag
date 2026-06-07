import sys
import os
import json
import time
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables (contains OPENAI_API_KEY)
load_dotenv(os.path.join(os.path.dirname(__file__), '../ablation_tests/.env'))

# Set default OLLAMA_HOST to localhost for local terminal execution
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

# Add backend to python path so we can import LightRAG and RAGMetrics
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'backend', 'src'))
sys.path.append(os.path.join(project_root, 'backend'))

from modules.light_retriever.light_retriever import LightRAGService
from modules.metrics.rag_metrics import RAGMetrics

MODELS = ["reduced", "base", "improved"]

async def run_capacity_evaluation_for_file(filepath: str, is_corporate: bool):
    print(f"\n=====================================================")
    print(f" Starting Capacity Evaluation for {os.path.basename(filepath)}")
    print(f"=====================================================")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        questions = json.load(f)
        
    capacity_tests_dir = os.path.dirname(__file__)
    filename_base = os.path.basename(filepath).replace(".json", "")
    output_intermediate = os.path.join(capacity_tests_dir, f"capacity_{filename_base}_results_intermediate.json")
    output_final = os.path.join(capacity_tests_dir, f"capacity_{filename_base}_results_final.json")
    
    cache_dir = "corporate_lightrag_cache" if is_corporate else "lightrag_cache"
    working_dir = os.path.join(project_root, 'data', cache_dir)
    
    # Define models configuration
    models_config = {
        "reduced": {
            "llm_model": "qwen2:1.5b",
            "working_dir": working_dir,
            "embed_model": "mxbai-embed-large",
            "chunk_size": 300,
            "chunk_overlap": 50
        },
        "base": {
            "llm_model": "phi3:mini",
            "working_dir": working_dir,
            "embed_model": "mxbai-embed-large",
            "chunk_size": 300,
            "chunk_overlap": 50
        },
        "improved": {
            "llm_model": "phi4-mini:3.8b",
            "working_dir": working_dir,
            "embed_model": "mxbai-embed-large",
            "chunk_size": 300,
            "chunk_overlap": 50
        }
    }
    
    # Initialize LightRAG services for each model
    services = {}
    for m_key, m_cfg in models_config.items():
        print(f"Initializing LightRAGService for {m_key} ({m_cfg['llm_model']})...")
        service = LightRAGService(m_cfg)
        await service.rag.initialize_storages()
        services[m_key] = service
        
    print("Initializing RAGMetrics Service (requires OPENAI_API_KEY for GPT-4o-mini)...")
    metrics_service = RAGMetrics()
    
    results = []
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
            "models": {}
        }
        
        for m_key in MODELS:
            print(f"  -> Running Model: {m_key} ({models_config[m_key]['llm_model']})")
            try:
                start_time = time.time()
                # We always evaluate using the mix mode of LSRAG
                search_results = await services[m_key].search(query, num=5, mode="mix")
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
                
                q_results["models"][m_key] = {
                    "answer": answer,
                    "response_time_seconds": response_time,
                    "metrics": metrics
                }
            except Exception as e:
                print(f"  -> Error in Model {m_key}: {e}")
                q_results["models"][m_key] = {
                    "answer": f"ERROR: {str(e)}",
                    "response_time_seconds": 0.0,
                    "metrics": {"faithfulness": 0.0, "recall": 0.0, "relevancy": 0.0, "ndcg": 0.0}
                }
                
        results.append(q_results)
        
        # Save intermediate after every question
        with open(output_intermediate, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    # Calculate Averages
    print("\n--- Calculating Averages ---")
    averages = {
        model: {
            "response_time_seconds": 0.0,
            "faithfulness": 0.0,
            "recall": 0.0,
            "relevancy": 0.0,
            "ndcg": 0.0
        } for model in MODELS
    }
    
    categories = list(set([r["category"] for r in results]))
    category_averages = {
        cat: {
            model: {
                "response_time_seconds": 0.0,
                "faithfulness": 0.0,
                "recall": 0.0,
                "relevancy": 0.0,
                "ndcg": 0.0
            } for model in MODELS
        } for cat in categories
    }
    category_counts = {cat: 0 for cat in categories}
    
    for r in results:
        cat = r["category"]
        category_counts[cat] += 1
        for model in MODELS:
            m_data = r["models"].get(model, {})
            resp_time = m_data.get("response_time_seconds", 0.0)
            mets = m_data.get("metrics", {})
            f_score = mets.get("faithfulness") or 0.0
            r_score = mets.get("recall") or 0.0
            rel_score = mets.get("relevancy") or 0.0
            n_score = mets.get("ndcg") or 0.0
            
            # Global averages
            averages[model]["response_time_seconds"] += resp_time
            averages[model]["faithfulness"] += f_score
            averages[model]["recall"] += r_score
            averages[model]["relevancy"] += rel_score
            averages[model]["ndcg"] += n_score
            
            # Category averages
            category_averages[cat][model]["response_time_seconds"] += resp_time
            category_averages[cat][model]["faithfulness"] += f_score
            category_averages[cat][model]["recall"] += r_score
            category_averages[cat][model]["relevancy"] += rel_score
            category_averages[cat][model]["ndcg"] += n_score
            
    num_questions = len(results)
    if num_questions > 0:
        for model in MODELS:
            for k in averages[model]:
                averages[model][k] /= num_questions
                
        for cat in categories:
            count = category_counts[cat]
            if count > 0:
                for model in MODELS:
                    for k in category_averages[cat][model]:
                        category_averages[cat][model][k] /= count
                        
    final_output = {
        "global_averages": averages,
        "category_averages": category_averages,
        "details": results
    }
    
    with open(output_final, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Capacity evaluation results saved to {output_final}")

async def main():
    print("=====================================================")
    print(" L S R A G   -   C A P A C I T Y   T E S T S")
    print("=====================================================\n")
    
    base_dir = os.path.dirname(__file__)
    ablation_tests_dir = os.path.abspath(os.path.join(base_dir, "../ablation_tests"))
    
    general_file = os.path.join(ablation_tests_dir, "evaluation_questions.json")
    corp_file = os.path.join(ablation_tests_dir, "evaluation_questions_corporate.json")
    
    if os.path.exists(general_file):
        await run_capacity_evaluation_for_file(general_file, is_corporate=False)
    else:
        print(f"File not found: {general_file}")
        
    if os.path.exists(corp_file):
        await run_capacity_evaluation_for_file(corp_file, is_corporate=True)
    else:
        print(f"File not found: {corp_file}")

if __name__ == "__main__":
    asyncio.run(main())
