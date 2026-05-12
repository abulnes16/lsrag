import os
import asyncio
from flashrag.retriever import BaseRetriever
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from typing import List, Dict


class LightRetriever(BaseRetriever):
    """
    Custom Retriever bridging FlashRAG and LightRAG using local Ollama models.
    """
    def __init__(self, config):
        super().__init__(config)
        
     
        settings = self.map_flashrag_to_lightrag(config)
        self.working_dir = settings["working_dir"]
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            
     
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=settings["llm_model"], 
            embedding_func=EmbeddingFunc(
                embedding_dim=settings["embed_dim"],  
                max_token_size=8192,
                func=lambda texts: ollama_embed(texts, embed_model=settings["embed_model"])
            ),
            chunk_token_size=settings["chunk_size"],
            chunk_overlap_token_size=settings["chunk_overlap"],
            # CPU/Local GPU Optimization
            llm_model_max_async=settings["max_async"],
            embedding_func_max_async=settings["max_async"] * 2,
            addon_params={"timeout": settings["timeout"]} 
        )
        
        self.query_mode = config.get("lightrag_mode", "hybrid")

    @staticmethod
    def map_flashrag_to_lightrag(config) -> dict:
        """
        Maps FlashRAG configuration keys to LightRAG specific parameters.
        """
        base_data_path = os.getenv("DATA_PATH", "/app/data")
        
        return {
    
            "working_dir": config.get("save_dir", os.path.join(base_data_path, "lightrag_cache")),
            "llm_model": config.get("generator_model", "llama3.2:3b"),
            "embed_model": config.get("retrieval_model", "mxbai-embed-large"),
            "embed_dim": config.get("embedding_dim", 1024),
            "chunk_size": config.get("chunk_size", 300),
            "chunk_overlap": config.get("chunk_overlap", 50),
            "timeout": config.get("timeout", 1200),
            "max_async": config.get("max_async", 1)
        }

    async def batch_search(self, query_list: List[str], num: int = 5) -> List[List[Dict]]:
        batch_results = []
        for query in query_list:
            # Use aquery (async) instead of query (sync)
            lightrag_output = await self.rag.aquery(
                query, 
                param=QueryParam(mode=self.query_mode)
            )
            
            batch_results.append([{
                "id": f"ollama_llama3_{self.query_mode}",
                "contents": lightrag_output,
                "score": 1.0
            }])
            
        return batch_results

    async def search(self, query: str, num: int = 5) -> List[Dict]:
        results = await self.batch_search([query], num)
        return results[0]