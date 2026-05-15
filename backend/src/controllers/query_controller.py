
from modules import LightRAGService 


class QueryController: 
    def __init__(self, light_service, naive_service):
        self.module_name = "[Query Controller]"
        self.light_service = light_service
        self.naive_service = naive_service

    async def process_query(self, query: str, rag_type: str = "lightrag", lightrag_mode: str = None):
        if rag_type == "naiverag":
            if not self.naive_service:
                raise ValueError("NaiveRAG service not initialized.")
            return await self.naive_service.search(query)
        else:
            if not self.light_service:
                raise ValueError("LightRAG service not initialized.")
            return await self.light_service.search(query, mode=lightrag_mode)
