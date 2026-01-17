import logging
import re
from typing import Dict, TypedDict, Tuple

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from src.config import ABUSIVE_WORDS_PATH, CURSE_WORDS_PATH

class CensorshipInput(TypedDict):
    query: str

class CensorshipOutput(BaseModel):
    is_confident: bool = Field(description="Флаг наличия запрещенной лексики")
    result: list[str] = Field(description="Обнаруженные запрещенные слова")

class CensorshipDetector:
    def __init__(
        self,
        abusive_words_path: str = ABUSIVE_WORDS_PATH,
        curse_words_path: str = CURSE_WORDS_PATH,
    ):
        self.banned_words = self._load_banned_words(abusive_words_path, curse_words_path)
        self.regex_pattern = self._build_regex_pattern()
    
    def _load_banned_words(self, *file_paths) -> set[str]:
        """Загружает запрещенные слова из файлов"""
        words = set()
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    words.update(line.strip().lower() for line in f if line.strip())
            except FileNotFoundError:
                logging.error(f"Файл не найден: {file_path}")
            except Exception as e:
                logging.error(f"Ошибка чтения файла {file_path}: {str(e)}")
        return words

    def _build_regex_pattern(self) -> re.Pattern:
        """Создает регулярное выражение для поиска целых слов"""
        escaped_words = [re.escape(word) for word in self.banned_words]
        pattern = r'(?<!\w)(' + '|'.join(escaped_words) + r')(?!\w)'
        return re.compile(pattern, re.IGNORECASE)

    def detect(self, text: str) -> Tuple[bool, list[str]]:
        """Обнаруживает запрещенные слова в тексте"""
        if not self.banned_words:
            logging.warning("Список запрещенных слов пуст")
            return False, []
        
        text = text.lower()
        matches = self.regex_pattern.findall(text)
        unique_matches = list(set(matches))
        
        return bool(unique_matches), unique_matches

def create_censorship_chain() -> Runnable[CensorshipInput, CensorshipOutput]:
    detector = CensorshipDetector()

    class CensorshipRunnable(Runnable[CensorshipInput, CensorshipOutput]):
        def invoke(self, input_data: CensorshipInput) -> CensorshipOutput:
            is_confident, result = detector.detect(input_data["query"])
            return CensorshipOutput(
                is_confident=is_confident,
                result=result
            )

    return CensorshipRunnable()