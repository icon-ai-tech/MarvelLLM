from abc import ABC, abstractmethod

from src.protection.models import ProtectionResult


class BaseProtector(ABC):

    @abstractmethod
    def check(self, query: str) -> ProtectionResult:
        """Check query"""


class BaseHandler(ABC):

    @abstractmethod
    async def ahandle_prompt(self, prompt: str, chat_id: str) -> str:
        """Handle prompt from user."""
