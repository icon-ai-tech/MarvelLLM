from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import (
    CONTEXTUALIZE_CHAIN_NODE_LLM_TEMPERATURE,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_URL_MODEL,
)
from src.nodes.contextualize.system_prompt import CONTEXTUALIZE_CHAIN_NODE_SYSTEM_PROMPT


class ContextualizeInput(TypedDict):
    input: str
    chat_history: List[BaseMessage]


class ContextualizeOutput(BaseModel):
    rephrased_query: str = Field(description="Переформулированный вопрос")


def create_contextualize_chain(
    llm_name: str = LLM_MODEL_NAME,
    temperature: float = CONTEXTUALIZE_CHAIN_NODE_LLM_TEMPERATURE,
    api_base: str = LLM_URL_MODEL,
    api_key: str = LLM_API_KEY,
) -> Runnable[ContextualizeInput, ContextualizeOutput]:
    prompt_template = PromptTemplate.from_template(
        """
        Системный промпт:
        {system_prompt}
        
        История соообщений:
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

    class ContextualizeRunnable(Runnable[ContextualizeInput, ContextualizeOutput]):
        def invoke(self, input_data: ContextualizeInput) -> ContextualizeOutput:
            prompt = prompt_template.format(
                system_prompt=CONTEXTUALIZE_CHAIN_NODE_SYSTEM_PROMPT,
                context=input_data["chat_history"],
                query=input_data["input"],
            )
            response = llm.invoke(prompt)
            print(response)
            # Возвращаем структурированный результат
            return ContextualizeOutput(rephrased_query=response.content.strip())

    return ContextualizeRunnable()
