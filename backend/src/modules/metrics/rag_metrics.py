from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, ContextRecall, AnswerRelevancy
import numpy as np
from sklearn.metrics import ndcg_score



class RAGMetrics:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.llm = llm_factory("gpt-4o-mini", client=self.client, max_tokens=8192)
        self.embeddings = embedding_factory("openai", model="text-embedding-3-small", client=self.client)
        self.faithfullness_scorer = Faithfulness(llm=self.llm)
        self.recall_scorer = ContextRecall(llm=self.llm)
        self.relevancy_scorer = AnswerRelevancy(llm=self.llm, embeddings = self.embeddings)
    
    def truncate_contexts(self, contexts: list[str], max_chars: int = 6000) -> list[str]:
        if not contexts:
            return contexts
        total_len = sum(len(c) for c in contexts)
        if total_len <= max_chars:
            return contexts
        
        truncated = []
        current_len = 0
        for c in contexts:
            if current_len >= max_chars:
                break
            available = max_chars - current_len
            if len(c) <= available:
                truncated.append(c)
                current_len += len(c)
            else:
                truncated_item = c[:available]
                last_space = truncated_item.rfind(' ')
                if last_space != -1:
                    truncated_item = truncated_item[:last_space]
                truncated.append(truncated_item + "...")
                break
        return truncated

    async def calculate_faithfulness(self, question: str, answer: str, contexts: list[str]):
        truncated = self.truncate_contexts(contexts)
        return await self.faithfullness_scorer.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=truncated
        )
    
    async def calculate_recall(self, question: str, reference: str, contexts: list[str]):
        truncated = self.truncate_contexts(contexts)
        return await self.recall_scorer.ascore(
           user_input=question,
           retrieved_contexts=truncated,
           reference=reference
        )
    
    async def calculate_relevancy(self, question: str, answer: str):
        return await self.relevancy_scorer.ascore(
            user_input=question,
            response=answer
        )
    
    async def calculate_ndcg(self, reference: str, contexts: list[str]) -> float:
        if not contexts or len(contexts) <= 1:
            return 0.0
        try:
            ref_embed = np.array(await self.embeddings.aembed_text(reference))
            context_embeds = [np.array(e) for e in await self.embeddings.aembed_texts(contexts)]
            
            y_true = []
            for c_emb in context_embeds:
                norm_c = np.linalg.norm(c_emb)
                norm_r = np.linalg.norm(ref_embed)
                if norm_c == 0 or norm_r == 0:
                    sim = 0.0
                else:
                    sim = np.dot(c_emb, ref_embed) / (norm_c * norm_r)
                y_true.append((sim + 1.0) / 2.0)
                
            n = len(contexts)
            y_score = [float(n - i) for i in range(n)]
            
            score = ndcg_score(np.array([y_true]), np.array([y_score]))
            return float(score)
        except Exception as e:
            print(f"Error calculating NDCG: {e}")
            return 0.0

    async def calculate_all_metrics(self, question: str, answer: str, contexts: list[str], reference: str):
        faithfulness = await self.calculate_faithfulness(question, answer, contexts)
        recall = await self.calculate_recall(question, reference, contexts)
        relevancy = await self.calculate_relevancy(question, answer)
        ndcg = await self.calculate_ndcg(reference, contexts)
        
        def to_float(val):
            if val is None:
                return 0.0
            if hasattr(val, "value"):
                return float(val.value) if val.value is not None else 0.0
            try:
                return float(val)
            except Exception:
                return 0.0

        return {
            "faithfulness": to_float(faithfulness),
            "recall": to_float(recall),
            "relevancy": to_float(relevancy),
            "ndcg": to_float(ndcg)
        }

        