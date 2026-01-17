from fastapi import Body, FastAPI

from src.config import LLM_MODEL_NAME, MONGO_DB_PATH
from src.handler import MarvelHandler, MarvelOptions

app = FastAPI()


def get_marvel_handler():
    options = MarvelOptions(
        llm_name=LLM_MODEL_NAME,
        psycopg_checkpointer=MONGO_DB_PATH,
    )
    return MarvelHandler(options)


handler = get_marvel_handler()

@app.post("/new_dialog")
async def new_dialog(user_id: str = Body(..., embed=True)):
    """
    Начинает новый диалог
    Пример тела запроса:
    {
        "user_id": "ваш_user_id"
    }
    """
    await handler.ahandle_prompt("начать новый диалог", user_id)
    return {"status": "dialog reset"}


@app.post("/handle_prompt")
async def handle_prompt(
    user_id: str = Body(...), prompt: str = Body(...), skip_faq: bool = Body(default=False, embed=True)
):
    """
    Обрабатывает запрос пользователя
    Пример тела запроса:
    {
        "user_id": "ваш_user_id",
        "prompt": "ваш запрос",
        "skip_faq": true/false
    }
    """
    result, flags = await handler.ahandle_prompt(prompt, user_id, skip_faq=skip_faq)
    return {
        "response": result,
        "is_faq_confident": flags["is_faq_confident"],
        "is_faq_minus_confident": flags["is_faq_minus_confident"],
        "faq_result": flags["faq_result"],
        "faq_minus_result": flags["faq_minus_result"],
        "faq_question": flags["faq_question"], 
    }
