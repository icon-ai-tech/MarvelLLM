import logging
from typing import Dict, Tuple, TypedDict, Union

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from pymilvus import Collection, connections

from src.config import MILVUS_HOST, MILVUS_PORT
from src.nodes.milvus.encoder import EmbeddingEncoder


class StopQuestionSearchInput(TypedDict):
    query: str


class StopQuestionSearchOutput(BaseModel):
    result: Dict[str, Union[str, float]] = Field(description="Найденный стоп-вопрос и ответ")
    is_confident: bool = Field(description="Флаг уверенности (score > 0.95)")


class StopQuestionSearcher:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        collection_name: str = "faq_minus_marvel",
        confidence_threshold: float = 0.95,
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
                "anns_field": "question_embedding",
                "param": {"metric_type": "COSINE", "params": {}},
                "limit": 1,
                "output_fields": ["stop_theme", "stop_questions_sample", "stop_answer"],
            }

            results = self.collection.search(**search_params)

            if not results or len(results[0]) == 0:
                return {}, False

            best_hit = results[0][0]
            result = {
                "stop_theme": best_hit.fields.get("stop_theme", ""),
                "stop_question": best_hit.fields.get("stop_questions_sample", ""),
                "stop_answer": best_hit.fields.get("stop_answer", ""),
                "score": float(best_hit.score),
            }

            is_confident = best_hit.score > self.confidence_threshold

            return result, is_confident

        except Exception as e:
            logging.error(f"Ошибка при поиске стоп-вопросов: {str(e)}")
            return {}, False


def create_stop_question_search_chain() -> Runnable[StopQuestionSearchInput, StopQuestionSearchOutput]:
    encoder = EmbeddingEncoder()
    searcher = StopQuestionSearcher(encoder=encoder)

    class StopQuestionSearchRunnable(Runnable[StopQuestionSearchInput, StopQuestionSearchOutput]):
        def invoke(self, input_data: StopQuestionSearchInput) -> StopQuestionSearchOutput:
            result, is_confident = searcher.search(input_data["query"])

            return StopQuestionSearchOutput(result=result, is_confident=is_confident)

    return StopQuestionSearchRunnable()
