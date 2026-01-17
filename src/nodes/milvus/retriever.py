import logging
import time
from typing import Dict, List, TypedDict, Union

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from pymilvus import Collection, MilvusException, connections

from src.config import MILVUS_HOST, MILVUS_PORT
from src.nodes.milvus.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)


class MilvusSearchInput(TypedDict):
    query: str


class MilvusSearchOutput(BaseModel):
    search_result: List[Dict[str, Union[str, int, float]]] = Field(description="Результаты поиска по документам")


class MilvusRetriever:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        host: str = MILVUS_HOST,
        port: int = MILVUS_PORT,
        connection_attempts: int = 3,
        retry_delay: int = 1,
    ):
        self.encoder = encoder
        self.host = host
        self.port = port
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay

    def _connect_to_milvus(self) -> bool:
        """Подключение к Milvus с повторными попытками"""
        for attempt in range(1, self.connection_attempts + 1):
            try:
                connections.connect("default", host=self.host, port=self.port)
                logger.info("Успешное подключение к Milvus")
                return True
            except MilvusException as e:
                logger.warning(f"Попытка подключения {attempt}/{self.connection_attempts} не удалась: {str(e)}")
                if attempt < self.connection_attempts:
                    time.sleep(self.retry_delay)
        logger.error("Не удалось подключиться к Milvus после всех попыток")
        return False

    def search(self, query: str) -> List[Dict[str, Union[str, int, float]]]:
        query_embedding = self.encoder.encode(query)
        if not query_embedding:
            return []

        if not self._connect_to_milvus():
            return []

        all_results = []
        try:
            collection = Collection("marvel_chunks_e5")
            collection.load()

            anns_field = "chunk_vec"  # ищем по maintext_vec
            search_params = {
                "data": [query_embedding],
                "param": {"metric_type": "COSINE", "params": {}},  # если используете IVF/HNSW – добавьте nprobe/ef
                "limit": 10,
                "output_fields": ["chunk_text", "source"],
            }

            results = collection.search(anns_field=anns_field, **search_params)

            try:
                sample = results[0][0]
                logger.info(f"Milvus entity keys: {list(sample.entity.keys())}")
            except Exception as e:
                logger.warning(f"Can't inspect entity keys: {e}")

            for hit in results[0]:
                entity = hit.entity
                title = entity.get("title", "")
                maintext = entity.get("chunk_text", "")
                url = entity.get("source", "")
                description = entity.get("comment", "")

                all_results.append(
                    {
                        "document_name": title,
                        "page_number": 0,
                        "header": description,
                        "text": maintext,
                        "url": url,
                        "score": float(hit.score),
                        "id": hit.id,
                    }
                )


            res = sorted(all_results, key=lambda x: x["score"], reverse=True)
            print(res)
            return res[:5]

        except MilvusException as e:
            logger.error(f"Milvus error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Критическая ошибка при выполнении поиска: {str(e)}")
            return []




def create_retriever_chain() -> Runnable[MilvusSearchInput, MilvusSearchOutput]:
    encoder = EmbeddingEncoder()
    retriever = MilvusRetriever(encoder=encoder)

    class MilvusSearchRunnable(Runnable[MilvusSearchInput, MilvusSearchOutput]):
        def invoke(self, input_data: MilvusSearchInput) -> MilvusSearchOutput:
            results = retriever.search(input_data["query"])
            return MilvusSearchOutput(search_result=results)

    return MilvusSearchRunnable()