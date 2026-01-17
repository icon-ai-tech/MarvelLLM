from dataclasses import dataclass

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from src.nodes.answer.answer import AnswerInput, create_answer_chain
from src.nodes.contextualize.contextualize import ContextualizeInput, create_contextualize_chain
from src.nodes.milvus.faq import FAQSearchInput, create_faq_search_chain
from src.nodes.milvus.faq_minus import StopQuestionSearchInput, create_stop_question_search_chain
from src.nodes.milvus.retriever import MilvusSearchInput, create_retriever_chain
from src.nodes.scenario.scenario import ScenarioInput, create_scenario_chain
from src.nodes.sql_nodes.sql_gen import SQLGenInput, create_sql_gen_chain
from src.nodes.censor.censor import CensorshipInput, create_censorship_chain


@dataclass
class MarvelRunnables:
    """
    Контейнер для всех Runnable, используемых в MarvelAssistant.
    """
    faq_minus_chain: Runnable[StopQuestionSearchInput, AIMessage]
    faq_chain: Runnable[FAQSearchInput, AIMessage]
    scenario_chain: Runnable[ScenarioInput, AIMessage]
    sql_gen_chain: Runnable[SQLGenInput, AIMessage]
    retriever_chain: Runnable[MilvusSearchInput, AIMessage]
    answer_chain: Runnable[AnswerInput, AIMessage]
    contextualize_chain: Runnable[ContextualizeInput, AIMessage]
    censorship_chain: Runnable[CensorshipInput, AIMessage]


def createMarvelRunnables(
    llm_name: str,
) -> MarvelRunnables:
    """
    Создаёт и возвращает набор Runnable для Marvel.

    Args:
        llm_name: Название модели LLM.
        headers: Заголовки для HTTP-запросов (необязательно).

    Returns:
        MarvelRunnables: Набор Runnable для MarvelAssistant.
    """
    
    censorship_chain = create_censorship_chain()
    faq_minus_chain = create_stop_question_search_chain()
    faq_chain = create_faq_search_chain()
    retriever_chain = create_retriever_chain()

    answer_chain = create_answer_chain(llm_name=llm_name)
    contextualize_chain = create_contextualize_chain(llm_name=llm_name)
    scenario_chain = create_scenario_chain(llm_name=llm_name)

    sql_gen_chain = create_sql_gen_chain(llm_name=llm_name)

    return MarvelRunnables(
        censorship_chain=censorship_chain,
        faq_minus_chain=faq_minus_chain,
        faq_chain=faq_chain,
        contextualize_chain=contextualize_chain,
        scenario_chain=scenario_chain,
        retriever_chain=retriever_chain,
        sql_gen_chain=sql_gen_chain,
        answer_chain=answer_chain,
    )
 