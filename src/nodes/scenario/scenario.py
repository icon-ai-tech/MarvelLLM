from typing import List, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import (
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_URL_MODEL,
    SCENARIO_NODE_LLM_TEMPERATURE,
)
from src.nodes.scenario.system_prompt import SCENARIO_CHAIN_NODE_SYSTEM_PROMPT


class ScenarioInput(TypedDict):
    query: str
    messages: List[BaseMessage]


class ScenarioOutput(BaseModel):
    action: str = Field(description="Тип действия: RAG, SQL или STUB")


def create_scenario_chain(
    llm_name: str = LLM_MODEL_NAME,
    temperature: float = SCENARIO_NODE_LLM_TEMPERATURE,
    api_base: str = LLM_URL_MODEL,
    api_key: str = LLM_API_KEY,
) -> Runnable[ScenarioInput, ScenarioOutput]:

    llm = ChatOpenAI(
        model=llm_name, temperature=temperature, openai_api_base=api_base, openai_api_key=api_key, max_retries=3, extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
    )

    prompt_template = PromptTemplate.from_template(
        """
        Системный промпт:
        {system_prompt}
        
        Запрос пользователя: 
        {query}
        """
    )

    class ScenarioRunnable(Runnable[ScenarioInput, ScenarioOutput]):
        def invoke(self, input_data: ScenarioInput) -> ScenarioOutput:
            try:
                prompt = prompt_template.format(
                    system_prompt=SCENARIO_CHAIN_NODE_SYSTEM_PROMPT, query=input_data["query"]
                )
                response = llm.invoke(prompt)
                action = response.content.strip().upper()
                print('action')
                print(action)
                return ScenarioOutput(action=action)

            except Exception:
                return ScenarioOutput(action="RAG")

    return ScenarioRunnable()
