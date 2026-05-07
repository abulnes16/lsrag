import os
import asyncio
from flashrag.retriever import BaseRetriever
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from typing import List, Dict


class LightRetriever(BaseRetriever):
    """
    Custom Retriever bridging FlashRAG and LightRAG using local Ollama models.
    """
    def __init__(self, config):
        super().__init__(config)
        
        base_data_path = os.getenv("DATA_PATH", "./data")
        self.working_dir = config.get("lightrag_working_dir", os.path.join(base_data_path, "lightrag_cache"))
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            
        # Initialize LightRAG natively with Ollama
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            # Use phi3:mini model as SLM to test the results
            llm_model_name='phi3:mini', 
            # Define embedding function
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,  
                max_token_size=8192,
                func=lambda texts: ollama_embedding(texts, embed_model="mxbai-embed-large")
            ),
            chunk_token_size=600,
            chunk_overlap_token_size=100
        )
        
        # Async initialization for local storage endpoints
        asyncio.run(self.rag.initialize_storages())
        
        self.query_mode = config.get("lightrag_mode", "hybrid")

    def batch_search(self, query_list: List[str], num: int = 5) -> List[List[Dict]]:
        batch_results = []
        for query in query_list:
            lightrag_output = self.rag.query(
                query, 
                param=QueryParam(mode=self.query_mode)
            )
            
            batch_results.append([{
                "id": f"ollama_phi3_{self.query_mode}",
                "contents": lightrag_output,
                "score": 1.0
            }])
            
        return batch_results

    def search(self, query: str, num: int = 5) -> List[Dict]:
        return self.batch_search([query], num)[0]