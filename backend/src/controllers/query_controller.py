
from modules import LightRAGService 


class QueryController: 
    def __init__(self, light_service, naive_service, metrics_service):
        self.module_name = "[Query Controller]"
        self.light_service = light_service
        self.naive_service = naive_service
        self.metrics_service = metrics_service

    async def process_query(self, query: str, rag_type: str = "lightrag", lightrag_mode: str = None):
        if rag_type == "naiverag":
            if not self.naive_service:
                raise ValueError("NaiveRAG service not initialized.")
            return await self.naive_service.search(query)
        else:
            if not self.light_service:
                raise ValueError("LightRAG service not initialized.")
            return await self.light_service.search(query, mode=lightrag_mode)
    
    async def evaluate_rag(self, query: str, reference: str, lightrag_mode: str = "mix"):
        naive_response = await self.naive_service.search(query)
        lightrag_response = await self.light_service.search(query, mode = lightrag_mode)
        lightrag_metrics = await self.metrics_service.calculate_all_metrics(
            question=query, 
            answer=lightrag_response[0]["contents"], 
            contexts=lightrag_response[0]["contexts"],
            reference=reference
        )
        naive_metrics = await self.metrics_service.calculate_all_metrics(
            question=query, 
            answer=naive_response[0]["contents"], 
            contexts=naive_response[0]["contexts"],
            reference=reference
        )
        return {
            "naive": naive_metrics,
            "lightrag": lightrag_metrics
        }
        
