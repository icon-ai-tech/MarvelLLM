# 🦸‍♂️MarvelLLM

<p align="center">
  <img src="logo.png" alt="AcademicLLM Logo" width="400" height="400">
</p>

<p align="center">
  <em>Интеллектуальный Telegram-бот с LLM для ответов на вопросы по вселенной Marvel</em>
</p>

**MarvelLLM** — интеллектуальный Telegram-бот на базе LLM (Large Language Model), созданный для ответов на вопросы по вселенной Marvel. Бот помогает разобраться в персонажах, событиях, фильмах, сериалах, комиксах и альтернативных вселенных, предоставляя точную и связную информацию в диалоговом формате.

---

## 🚀 Telegram-бот по вселенной Marvel

@MarvelLLM_bot

---

## 🚀 Возможности

- 🦸‍♂️ Ответы на естественном языке на вопросы о вселенной Marvel
- 🌌 Информация о персонажах, командах, артефактах и событиях
- 🎬 Разбор фильмов, сериалов и хронологии MCU
- 📖 Справка по комиксам, альтернативным вселенным и сюжетным аркам
- 🔎 Интеллектуальный поиск знаний с использованием RAG (retrieval-augmented generation) для точных и связных ответов

---

## 🛠️ Технологии

- [Python 3.12.2](https://www.python.org/)
- [Poetry](https://python-poetry.org/)
- LLM (OpenAI / Mistral / Llama)
- Retrieval-Augmented Generation (RAG)
- FastAPI (для API сервиса)
- Uvicorn (ASGI сервер)

---

## 📦 Установка и запуск

1. **Подготовка данных**:

   - Поместите данные в папку `data/` проекта
   - Настройте пути в `src/config.py`
2. **Установка зависимостей**:

2.1

```bash
git clone https://github.com/icon-ai-tech/MarvelLLM.git
cd MarvelLLM
```

2.2

```bash
poetry install
poetry env activate
source /home/user/.cache/pypoetry/virtualenvs/marvelllm-XXX-XXX-py3.12/bin/activate
```

3. **Запуск сервисов**:

В разных терминалах выполните:

```bash
# Запуск API сервиса
uvicorn src.api_service:app --host 0.0.0.0 --port 8012

# Запуск Telegram бота
python src/bot.py
```

Для разработки можно добавить `-v $(pwd)/src:/app/src` чтобы сразу видеть изменения кода.
