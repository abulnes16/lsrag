from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
import sys
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Ensure 'src' is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.data_ingestor import DataIngestor
from modules import LightRetriever

from flashrag.config import Config

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing LightRetriever...")
    
    
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
    
    await retriever.rag.initialize_storages()
    
    app.state.retriever = retriever
    
    print("Starting up: Running Data Ingestion...")
    ingestor = DataIngestor(retriever)
    await ingestor.ingest_datasets(sample_size=0.001)
    
   
    
    yield
    
    print("Shutting down...")



app = FastAPI(title="LSRAG API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

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
    try:
        # Use our custom RAG retriever!
        retriever = app.state.retriever
        results = await retriever.search(request.query)
        
        # The first result's contents will have the LightRAG generated answer
        answer = results[0]["contents"]
        
        return {"response": answer}
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return {"error": str(e)}
