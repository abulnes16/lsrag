from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    rag_type: str = "lightrag" # 'lightrag' or 'naiverag'
    lightrag_mode: Optional[str] = None

class EvaluateRequest(BaseModel):
    query: str
    reference: str
    lightrag_mode: Optional[str] = "mix"
