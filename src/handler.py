from typing import Any, Dict, Optional

from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel
from pymongo import MongoClient

from src.assistant_graph import MarvelAssistant
from src.config import MAX_LEN_USER_PROMPT
from src.protection import ExceedingProtector, ProtectionStatus, ProtectorsAccumulator
from src.protection.base import BaseHandler
from src.runnables import createMarvelRunnables


class MarvelOptions(BaseModel):
    llm_name: str
    psycopg_checkpointer: str


class MarvelHandler(BaseHandler):

    def __init__(self, options: MarvelOptions) -> None:
        self._marvel_runnables = createMarvelRunnables(llm_name=options.llm_name)
        self._checkpointer_db_uri = options.psycopg_checkpointer
        self._protector = ProtectorsAccumulator(protectors=[ExceedingProtector(max_len=MAX_LEN_USER_PROMPT)])
        mongodb_client = MongoClient(self._checkpointer_db_uri)
        checkpointer = MongoDBSaver(mongodb_client)
        self.assistant = MarvelAssistant(
            runnables=self._marvel_runnables,
            checkpointer=checkpointer,
        )

    async def ahandle_prompt(
        self, prompt: str, chat_id: str, skip_faq: Optional[bool] = False
    ) -> tuple[str, Dict[str, Any]]:
        """
        Обрабатывает запрос пользователя с возможностью пропуска FAQ проверки

        Args:
            prompt: Текст запроса пользователя
            chat_id: Идентификатор чата
            skip_faq: Флаг пропуска проверки FAQ (по умолчанию False)

        Returns:
            Кортеж (ответ, словарь с флагами и результатами проверок)
        """
        protector_res = self._protector.check(prompt)
        if protector_res.status is not ProtectionStatus.ok:
            return protector_res.message, {
                "is_faq_confident": False,
                "is_faq_minus_confident": False,
                "faq_result": None,
                "faq_minus_result": None,
            }

        config = {"configurable": {"thread_id": chat_id, "skip_faq": skip_faq}}

        output = self.assistant.graph.invoke({"query": prompt, "user_id": chat_id}, config=config)

        answer = output["final_output"]

        faq_result = output.get("faq_result")
        faq_question = None
        if faq_result:
            faq_question = faq_result.get("header")

        return answer, {
            "is_faq_confident": output.get("is_faq_confident", False),
            "is_faq_minus_confident": output.get("is_faq_minus_confident", False),
            "faq_result": faq_result,
            "faq_minus_result": output.get("faq_minus_result"),
            "faq_question": faq_question
        }
