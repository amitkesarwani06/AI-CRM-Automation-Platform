from abc import ABC, abstractmethod
from app.core.chat_model import ChatModel, get_chat_model


class BaseAgent(ABC):
    """Every agent in the CRM platform inherits from this."""

    def __init__(self, name: str, system_prompt: str, chat_model: ChatModel | None = None):
        self.name = name
        self.system_prompt = system_prompt
        self.chat_model = chat_model or get_chat_model()  # injected, or default

    @abstractmethod
    def run(self, user_input: str) -> str:
        """Every subclass MUST implement this."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Used by the Supervisor to know what this agent does."""
        ...

    def _call_llm(self, user_input: str) -> str:
        """Shared plumbing — builds the messages and calls the ChatModel."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.chat_model.chat(messages)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
