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
from modules import LightRAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing LightRAGService...")
    
    config = {
        "working_dir": os.path.join(os.getenv("DATA_PATH", "/app/data"), "lightrag_cache"),
        "llm_model": "llama3.2:3b",
        "embed_model": "mxbai-embed-large",
        "embed_dim": 1024,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "timeout": 1200,
        "max_async": 1,
        "lightrag_mode": "hybrid"
    }
    
    retriever = LightRAGService(config)
    
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
