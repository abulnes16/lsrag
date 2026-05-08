
from modules import LightRetriever


class QueryController: 
    def __init__(self):
        self.module_name = "[Query Controller]"
        self.retriever_manager = LightRetriever()

    def process_query(self, query: str):
        return self.retriever_manager.search(query)
