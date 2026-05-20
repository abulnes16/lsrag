import os
import uuid
import ollama
import numpy as np
from typing import List, Dict
from nano_vectordb import NanoVectorDB

class NaiveRAGService:
    """
    Standalone Naive RAG Service for baseline comparison.
    Uses official Ollama library and NanoVectorDB.
    """
    def __init__(self, config: dict):
        base_data_path = os.getenv("DATA_PATH", "/app/data")
        self.working_dir = config.get("working_dir", os.path.join(base_data_path, "naive_data"))
        
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            
        self.db_path = os.path.join(self.working_dir, "nano_vectordb.json")
        self.llm_model = config.get("llm_model", "phi3:mini")
        self.embed_model = config.get("embed_model", "mxbai-embed-large")
        self.embed_dim = config.get("embed_dim", 1024)
        
        self.chunk_size = config.get("chunk_size", 300)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        
        # Initialize VectorDB
        self.vdb = NanoVectorDB(self.embed_dim, storage_file=self.db_path)
        
        # Determine host for Ollama client based on env (defaults to localhost outside docker)
        ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        self.client = ollama.AsyncClient(host=ollama_host)
        
    def chunk_text(self, text: str) -> List[str]:
        """Simple word-based chunking mimicking LightRAG's parameters."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += (self.chunk_size - self.chunk_overlap)
        return chunks

    async def initialize(self, texts: List[str]):
        """Chunks texts, computes embeddings, and stores in NanoVectorDB."""
        print(f"[NaiveRAG] Initializing with {len(texts)} documents...")
        all_chunks = []
        for text in texts:
            all_chunks.extend(self.chunk_text(text))
            
        print(f"[NaiveRAG] Created {len(all_chunks)} chunks. Computing embeddings...")
        
        upserts = []
        for i, chunk in enumerate(all_chunks):
            response = await self.client.embeddings(model=self.embed_model, prompt=chunk)
            embedding = np.array(response['embedding'])
            
            upserts.append({
                "__id__": str(uuid.uuid4()),
                "__vector__": embedding,
                "text": chunk
            })
            
            if (i + 1) % 100 == 0:
                print(f"[NaiveRAG] Embedded {i+1}/{len(all_chunks)} chunks...")
                
        # Upsert in bulk to NanoVectorDB
        self.vdb.upsert(upserts)
        self.vdb.save()
        print(f"[NaiveRAG] Initialization complete. Stored {len(upserts)} chunks in NanoVectorDB.")

    async def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Embeds query and retrieves top K chunks from NanoVectorDB."""
        response = await self.client.embeddings(model=self.embed_model, prompt=query)
        query_embedding = np.array(response['embedding'])
        
        results = self.vdb.query(query_embedding, top_k=top_k)
        
        contexts = [res['text'] for res in results]
        return contexts

    async def generate(self, contexts: List[str], query: str) -> str:
        """Generates the final answer using Ollama API directly."""
        context_str = "\n\n".join(contexts)
        prompt = f"""
        You are a helpful assistant answering questions based strictly on the provided context.
        Context:
        {context_str}
        Question: {query}
        """
        
        response = await self.client.chat(model=self.llm_model, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])
        return response['message']['content']

    async def batch_search(self, query_list: List[str], num: int = 5) -> List[List[Dict]]:
        """Compatible interface with LightRAGService."""
        batch_results = []
        for query in query_list:
            contexts = await self.retrieve(query, top_k=num)
            answer = await self.generate(contexts, query)
            
            batch_results.append([{
                "id": "naive_rag_phi3",
                "contents": answer,
                "contexts": contexts
            }])
        return batch_results

    async def search(self, query: str, num: int = 5) -> List[Dict]:
        """Compatible interface with LightRAGService."""
        results = await self.batch_search([query], num)
        return results[0]
