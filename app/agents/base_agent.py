from abc import ABC, abstractmethod
from app.core.llm_client import ask_llm
from app.core.config import LLMConfig, DEFAULT_CONFIG


class BaseAgent(ABC):
    """Every agent in the CRM platform inherits from this."""

    def __init__(self, name: str, system_prompt: str, config: LLMConfig = DEFAULT_CONFIG):
        self.name = name
        self.system_prompt = system_prompt
        self.config = config
    
    @abstractmethod
    def run(self, user_input: str) -> str:
        """Every subclass MUST implement this."""
        raise NotImplementedError
    @property                          # <-- Makes it accessible as agent.description (no parentheses)
    @abstractmethod                    # <-- Forces EVERY child class to implement it
    def description(self) -> str:
        """Used by the Supervisor to know what this agent does."""
        ...                            # <-- No implementation here — children provide it
    def _call_llm(self, user_input: str) -> str:
        """Shared plumbing — every subclass reuses this."""
        return ask_llm(user_input, system_prompt=self.system_prompt)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
