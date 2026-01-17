from typing import TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import (
    MAX_LLM_CONTEXT,
    ANSWER_NODE_LLM_TEMPERATURE,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_URL_MODEL,
)
from src.nodes.answer.system_prompt_rag import ANSWER_NODE_SYSTEM_PROMPT_RAG
from src.nodes.answer.system_prompt_sql import ANSWER_NODE_SYSTEM_PROMPT_SQL

class AnswerInput(TypedDict):
    context: str
    query: str
    scenario: str


class AnswerOutput(BaseModel):
    final_output: str = Field(description="Финальный ответ пользователю")


def create_answer_chain(
    llm_name: str = LLM_MODEL_NAME,
    temperature: float = ANSWER_NODE_LLM_TEMPERATURE,
    api_base: str = LLM_URL_MODEL,
    api_key: str = LLM_API_KEY,
) -> Runnable[AnswerInput, AnswerOutput]:
    prompt_template = PromptTemplate.from_template(
        """
        Системный промпт:
        {system_prompt}
        
        Контекст:
        {context}
        
        
        Пользовательский запрос:
        {query}
        """
    )

    llm = ChatOpenAI(
        model=llm_name, temperature=temperature, openai_api_base=api_base, openai_api_key=api_key, max_retries=3, extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
    )

    class AnswerRunnable(Runnable[AnswerInput, AnswerOutput]):
        def invoke(self, input_data: AnswerInput) -> AnswerOutput:
            if input_data["scenario"].action == "SQL":
                ANSWER_NODE_SYSTEM_PROMPT = ANSWER_NODE_SYSTEM_PROMPT_SQL
            else:
                ANSWER_NODE_SYSTEM_PROMPT = ANSWER_NODE_SYSTEM_PROMPT_RAG

            prompt = prompt_template.format(
                system_prompt=ANSWER_NODE_SYSTEM_PROMPT, context=input_data["context"][:MAX_LLM_CONTEXT], query=input_data["query"]
            )
            response = llm.invoke(prompt)
            result = response.content.strip()
            return AnswerOutput(final_output=result)

    return AnswerRunnable()
