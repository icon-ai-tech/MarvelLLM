import logging
from typing import Dict, Tuple, TypedDict, Union

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from pymilvus import Collection, connections

from src.config import CONFIDENCE_THRESHOLD, MILVUS_HOST, MILVUS_PORT
from src.nodes.milvus.encoder import EmbeddingEncoder


class FAQSearchInput(TypedDict):
    query: str


class FAQSearchOutput(BaseModel):
    result: Dict[str, Union[str, float]] = Field(description="Найденный FAQ-ответ")
    is_confident: bool = Field(description="Флаг уверенности (score > 0.95)")


class FAQSearcher:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        collection_name: str = "faq_for_rag_e5_marvel",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.encoder = encoder
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.confidence_threshold = confidence_threshold
        self._connect()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
        except Exception as e:
            logging.error(f"Ошибка подключения к Milvus: {str(e)}")
            raise

    def search(self, query: str) -> Tuple[Dict[str, Union[str, float]], bool]:
        try:
            embedding = self.encoder.encode(query)
            if not embedding:
                return {}, False

            search_params = {
                "data": [embedding],
                "anns_field": "qestion_e5",
                "param": {"metric_type": "COSINE", "params": {}},
                "limit": 1,
                "output_fields": ["text", "header", "question"],
            }

            results = self.collection.search(**search_params)

            if not results or len(results[0]) == 0:
                return {}, False

            best_hit = results[0][0]
            result = {
                "text": best_hit.fields.get("text", ""),
                "header": best_hit.fields.get("question") or best_hit.fields.get("header", ""),
                "score": float(best_hit.score),
            }

            is_confident = best_hit.score > self.confidence_threshold

            return result, is_confident

        except Exception as e:
            logging.error(f"Ошибка при поиске FAQ: {str(e)}")
            return {}, False


def create_faq_search_chain() -> Runnable[FAQSearchInput, FAQSearchOutput]:
    encoder = EmbeddingEncoder()
    searcher = FAQSearcher(encoder=encoder)

    class FAQSearchRunnable(Runnable[FAQSearchInput, FAQSearchOutput]):
        def invoke(self, input_data: FAQSearchInput) -> FAQSearchOutput:
            result, is_confident = searcher.search(input_data["query"])

            return FAQSearchOutput(result=result, is_confident=is_confident)

    return FAQSearchRunnable()
