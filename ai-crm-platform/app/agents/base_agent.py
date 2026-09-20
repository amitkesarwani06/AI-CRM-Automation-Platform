from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate
from app.core.chat_model import ChatModel, get_chat_model


class BaseAgent(ABC):
    """Every agent in the CRM platform inherits from this."""

    def __init__(self, name: str, prompt: ChatPromptTemplate, chat_model: ChatModel | None = None):
        self.name = name
        self.prompt = prompt
        self.chat_model = chat_model or get_chat_model()  # injected, or default

    @abstractmethod
    def run(self, user_input: str, **prompt_vars) -> str:
        """Every subclass MUST implement this."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Used by the Supervisor to know what this agent does."""
        ...

    def _call_llm(self, user_input: str, **prompt_vars) -> str:
        """Shared plumbing — builds the messages and calls the ChatModel."""
        # 1. Format the template variables using LangChain
        messages = self.prompt.format_messages(input=user_input, **prompt_vars)
        
        # 2. Map LangChain message types ("human" -> "user", "ai" -> "assistant")
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        as_dicts = [
            {"role": role_map.get(m.type, m.type), "content": m.content}
            for m in messages
        ]

        # 3. Send to ChatModel
        return self.chat_model.chat(as_dicts)


    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
