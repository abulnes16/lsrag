from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
import sys
from contextlib import asynccontextmanager

# Ensure 'src' is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.data_ingestor import DataIngestor
from modules import LightRetriever

from flashrag.config import Config

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing LightRetriever...")
    
    # Since FlashRAG's package is missing basic_config.yaml, we pass a raw dict
    # with all the necessary default parameters that BaseRetriever expects.
    config = {
        'retrieval_method': 'semantic',
        'retrieval_topk': 5,
        'retrieval_batch_size': 256,
        'retrieval_use_fp16': False,
        'retrieval_query_max_length': 128,
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'faiss_gpu': False,
        'use_sentence_transformer': False,
        'index_path': None,
        'corpus_path': None,
        'retrieval_model_path': None,
        'retrieval_pooling_method': 'mean',
        'model2path': {},
        'model2pooling': {},
        'method2index': {},
        'use_reranker': False,
    }
    
    retriever = LightRetriever(config)
    
    # Initialize storages asynchronously here since lifespan is already an async context
    await retriever.rag.initialize_storages()
    
    app.state.retriever = retriever
    
    print("Starting up: Running Data Ingestion...")
    ingestor = DataIngestor(retriever)
    await ingestor.ingest_datasets(sample_size=0.001)
    
   
    
    yield
    
    print("Shutting down...")

app = FastAPI(title="LSRAG API", lifespan=lifespan)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to LSRAG API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: QueryRequest):
    # Esto es un endpoint inicial que pasa la consulta a Ollama directamente.
    # En el futuro aquí se integrará la lógica RAG con FlashRAG, LightRAG, NanoVectorDB y NetworkX.
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": "phi3:mini",
                    "prompt": request.query,
                    "stream": False
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return {"response": data.get("response", "")}
        except Exception as e:
            return {"error": str(e)}
