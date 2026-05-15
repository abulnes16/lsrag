import os
import asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from typing import List, Dict


class LightRAGService:
    """
    Standalone Service for LightRAG using local Ollama models.
    """
    def __init__(self, config: dict):
        self.working_dir = config.get("working_dir", os.path.join(os.getenv("DATA_PATH", "/app/data"), "lightrag_cache"))
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=config.get("llm_model", "phi3:mini"), 
            embedding_func=EmbeddingFunc(
                embedding_dim=config.get("embed_dim", 1024),  
                max_token_size=8192,
                func=lambda texts: ollama_embed(texts, embed_model=config.get("embed_model", "mxbai-embed-large"))
            ),
            chunk_token_size=config.get("chunk_size", 300),
            chunk_overlap_token_size=config.get("chunk_overlap", 50),
            # CPU/Local GPU Optimization
            llm_model_max_async=config.get("max_async", 1),
            embedding_func_max_async=config.get("max_async", 1) * 2,
            addon_params={"timeout": config.get("timeout", 1200)} 
        )
        
        self.query_mode = config.get("lightrag_mode", "mix")

    async def batch_search(self, query_list: List[str], num: int = 5) -> List[List[Dict]]:
        batch_results = []
        for query in query_list:
            # Use aquery (async) instead of query (sync)
            lightrag_output = await self.rag.aquery(
                query, 
                param=QueryParam(mode=self.query_mode)
            )
            
            batch_results.append([{
                "id": f"ollama_phi3mini_{self.query_mode}",
                "contents": lightrag_output,
                "score": 1.0
            }])
            
        return batch_results

    async def search(self, query: str, num: int = 5) -> List[Dict]:
        results = await self.batch_search([query], num)
        return results[0]