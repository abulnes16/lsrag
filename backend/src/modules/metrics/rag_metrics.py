from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, ContextRecall, AnswerRelevancy



class RAGMetrics:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.llm = llm_factory("gpt-4o-mini", client=self.client)
        self.embeddings = embedding_factory("openai", model="text-embedding-3-small", client=self.client)
        self.faithfullness_scorer = Faithfulness(llm=self.llm)
        self.recall_scorer = ContextRecall(llm=self.llm)
        self.relevancy_scorer = AnswerRelevancy(llm=self.llm, embeddings = self.embeddings)
    
    async def calculate_faithfulness(self, question: str, answer: str, contexts: list[str]):
        return await self.faithfullness_scorer.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )
    
    async def calculate_recall(self, question: str, reference: str, contexts: list[str]):
        return await self.recall_scorer.ascore(
           user_input=question,
           retrieved_contexts=contexts,
           reference=reference
        )
    
    async def calculate_relevancy(self, question: str, answer: str):
        return await self.relevancy_scorer.ascore(
            user_input=question,
            response=answer
        )
    
    async def calculate_all_metrics(self, question: str, answer: str, contexts: list[str], reference: str):
        faithfulness = await self.calculate_faithfulness(question, answer, contexts)
        recall = await self.calculate_recall(question, reference, contexts)
        relevancy = await self.calculate_relevancy(question, answer)
        return {
            "faithfulness": faithfulness,
            "recall": recall,
            "relevancy": relevancy
        }
        