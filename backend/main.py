from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="L-SRAG API")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to L-SRAG API"}

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
