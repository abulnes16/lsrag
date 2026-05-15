
from modules import LightRAGService 


class QueryController: 
    def __init__(self):
        self.module_name = "[Query Controller]"
        self.retriever_manager = LightRAGService()

    def process_query(self, query: str):
        return self.retriever_manager.search(query)
