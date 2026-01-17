# Базовый образ с Python 3.12
FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN pip install --upgrade pip poetry

# Копируем только необходимые файлы для установки зависимостей
COPY pyproject.toml poetry.lock /app/

# Устанавливаем зависимости Python
WORKDIR /app
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

# Копируем остальные файлы проекта
COPY . .

# Команда для запуска обоих сервисов через waitress (можно заменить на другой процесс-менеджер)
CMD bash -c "uvicorn src.api_service:app --host 0.0.0.0 --port 8000 & python src/bot.py"