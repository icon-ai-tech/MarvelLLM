import json
import re
import sqlite3
from ast import literal_eval
from typing import List, TypedDict

import sqlparse
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from src.config import (
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_URL_MODEL,
    SQL_GEN_NODE_DB_PATH,
    SQL_GEN_NODE_LLM_TEMPERATURE,
)
from src.nodes.sql_nodes.system_prompt import SQL_SYSTEM_PROMPT


class SQLGenInput(TypedDict):
    query: str
    sql_shots: List[str]
    example_questions: List[str]


class SQLGenOutput(BaseModel):
    generated_sql: str = Field(description="Сгенерированный SQL-запрос и результаты его выполнения")


def create_sql_gen_chain(
    llm_name: str = LLM_MODEL_NAME,
    temperature: float = SQL_GEN_NODE_LLM_TEMPERATURE,
    db_path: str = SQL_GEN_NODE_DB_PATH,
    api_base: str = LLM_URL_MODEL,
    api_key: str = LLM_API_KEY,
) -> Runnable[SQLGenInput, SQLGenOutput]:

    llm = ChatOpenAI(
        model=llm_name,
        temperature=temperature,
        openai_api_base=api_base,
        openai_api_key=api_key,
        max_retries=3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    db = create_engine(db_path)

    prompt_template = PromptTemplate.from_template(
        """
        {system_prompt}

        Вопрос: 
        {query}
        """
    )

    class SQLGenRunnable(Runnable[SQLGenInput, SQLGenOutput]):
        def invoke(self, input_data: SQLGenInput) -> SQLGenOutput:

            prompt = prompt_template.format(
                system_prompt=SQL_SYSTEM_PROMPT,
                query=input_data["query"].lower(),
            )

            # Генерация SQL
            sql_response = llm.invoke(prompt)
            raw_sql = sql_response.content.strip()
            print('raw_sql')
            print(raw_sql)

            # 1. Очищаем возможные markdown-блоки
            cleaned = raw_sql.replace("```sql", "").replace("```", "").strip()

            # 2. Пытаемся распарсить через sqlparse
            try:
                statements = sqlparse.split(cleaned)
                valid_stmts = [
                    stmt.strip() for stmt in statements
                    if stmt.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE"))
                ]
                if valid_stmts:
                    generated_sql = valid_stmts[0]
                else:
                    raise ValueError("sqlparse не нашёл подходящих запросов")
            except Exception:
                # Фоллбек на regex
                pattern = re.compile(
                    r"(?:^|\s*)(SELECT|INSERT|UPDATE|DELETE)\b[\s\S]+?;",
                    re.IGNORECASE,
                )
                match = pattern.search(cleaned)
                generated_sql = match.group(0).strip() if match else cleaned

            # Убираем лишние ``` и пробелы
            generated_sql = generated_sql.replace("```", "").strip()
            if not generated_sql.endswith(";"):
                generated_sql += ";"

            # Выполнение запроса
            try:
                with db.connect() as conn:
                    result_proxy = conn.execute(text(generated_sql))
                    columns = result_proxy.keys()
                    rows = result_proxy.fetchall()

                    db_result = []
                    for row in rows:
                        row_dict = {}
                        for idx, col in enumerate(columns):
                            val = row[idx]
                            if isinstance(val, str) and val.strip().startswith("{"):
                                try:
                                    val = literal_eval(val)
                                except Exception:
                                    pass
                            row_dict[col] = val
                        db_result.append(row_dict)

                output_text = (
                    f"Запрос: {input_data['query']}\n\n"
                    f"Сгенерированный SQL:\n{generated_sql}\n\n"
                    f"Результаты:\n{db_result if db_result else 'Ничего не найдено'}"
                )
            except Exception as e:
                output_text = (
                    f"Запрос: {input_data['query']}\n\n"
                    f"Сгенерированный SQL:\n{generated_sql}\n\n"
                    f"Ошибка выполнения:\n{str(e)}"
                )

            return SQLGenOutput(generated_sql=output_text)

    return SQLGenRunnable()
