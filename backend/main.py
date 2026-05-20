from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import httpx
import os
import sys
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Ensure 'src' is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.data_ingestor import DataIngestor
from modules import LightRAGService, NaiveRAGService
from modules.metrics.rag_metrics import RAGMetrics
from controllers.query_controller import QueryController

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing RAG Services...")
    
    lightrag_config = {
        "working_dir": os.path.join(os.getenv("DATA_PATH", "/app/data"), "lightrag_cache"),
        "llm_model": "phi3:mini",
        "embed_model": "mxbai-embed-large",
        "embed_dim": 1024,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "timeout": 1200,
        "max_async": 1,
        "lightrag_mode": "hybrid"
    }
    
    naive_config = {
        "working_dir": os.path.join(os.getenv("DATA_PATH", "/app/data"), "naive_data"),
        "llm_model": "phi3:mini",
        "embed_model": "mxbai-embed-large",
        "embed_dim": 1024,
        "chunk_size": 300,
        "chunk_overlap": 50,
    }
    
    # Initialize both services
    light_service = LightRAGService(lightrag_config)
    naive_service = NaiveRAGService(naive_config)
    
    await light_service.rag.initialize_storages()
    
    # Store services and controller in app state
    app.state.light_service = light_service
    app.state.naive_service = naive_service
    metrics_service = RAGMetrics()
    app.state.query_controller = QueryController(light_service, naive_service, metrics_service)
    
    print("Starting up: Running Data Ingestion for both systems...")
    ingestor = DataIngestor(light_service, naive_service)
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
    rag_type: str = "lightrag" # 'lightrag' or 'naiverag'
    lightrag_mode: Optional[str] = None

class EvaluateRequest(BaseModel):
    query: str
    reference: str
    lightrag_mode: Optional[str] = "mix"

@app.get("/")
def read_root():
    return {"message": "Welcome to LSRAG API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        controller = app.state.query_controller
        results = await controller.process_query(request.query, request.rag_type, request.lightrag_mode)
        
        # The output format for both services is a list of dicts with a 'contents' field
        answer = results[0]["contents"]
        
        return {"response": answer}
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return {"error": str(e)}

@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    try:
        controller = app.state.query_controller
        results = await controller.evaluate_rag(request.query, request.reference, request.lightrag_mode)
        return results
    except Exception as e:
        print(f"Error in evaluate endpoint: {e}")
        return {"error": str(e)}
