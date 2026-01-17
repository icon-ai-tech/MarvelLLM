import os
from pathlib import Path
from typing import Final

from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ.update(config)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# === MongoDB ===
MONGO_USERNAME = os.getenv("MONGO_USERNAME", "")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
MONGO_HOST = os.getenv("MONGO_HOST", "")
MONGO_PORT = os.getenv("MONGO_PORT", "")
MONGO_DB_PATH = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "")
MAX_MESSAGES_HISTORY = int(os.getenv("MAX_MESSAGES_HISTORY", ""))

# === Milvus ===
MILVUS_HOST = os.getenv("MILVUS_HOST", "")
MILVUS_PORT = os.getenv("MILVUS_PORT", "")
CONFIDENCE_THRESHOLD = os.getenv("CONFIDENCE_THRESHOLD", 0.9)

# === Telegram ===
TOKEN = os.getenv("TOKEN", "")
MAX_LEN_USER_PROMPT = int(os.getenv("MAX_LEN_USER_PROMPT", 512))
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "")
HANDLER_API_URL = os.getenv("HANDLER_API_URL", "")
MAX_FAQ_MINUS_ATTEMPTS = int(os.getenv("MAX_FAQ_MINUS_ATTEMPTS", "30"))

# === API ===
LLM_URL_MODEL: Final[str] = os.getenv("LLM_URL_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")

# === Embedding ===
TEI_URL_EMBEDDER = os.getenv("TEI_URL_EMBEDDER", "")
TEI_API_KEY = os.getenv("TEI_API_KEY", "")
TEI_MODEL_NAME = os.getenv("TEI_MODEL_NAME", "")

# === LLM Configs ===
ANSWER_NODE_LLM_TEMPERATURE = float(os.getenv("ANSWER_NODE_LLM_TEMPERATURE", 0.7))
CONTEXTUALIZE_CHAIN_NODE_LLM_TEMPERATURE = float(os.getenv("CONTEXTUALIZE_CHAIN_NODE_LLM_TEMPERATURE", 0))
SCENARIO_NODE_LLM_TEMPERATURE = float(os.getenv("SCENARIO_NODE_LLM_TEMPERATURE", 0))
SQL_GEN_NODE_LLM_TEMPERATURE = float(os.getenv("SQL_GEN_NODE_LLM_TEMPERATURE", 0))
MAX_LLM_CONTEXT = int(os.getenv("MAX_LLM_CONTEXT", 24000))

# === Paths ===
SQL_GEN_NODE_DB_PATH = os.getenv("SQL_GEN_NODE_DB_PATH", f"sqlite:///{DATA_DIR / 'marvel_characters.db'}")
ABUSIVE_WORDS_PATH = os.getenv("ABUSIVE_WORDS_PATH", str(DATA_DIR / "ru_abusive_words.txt"))
CURSE_WORDS_PATH = os.getenv("CURSE_WORDS_PATH", str(DATA_DIR / "ru_curse_words.txt"))