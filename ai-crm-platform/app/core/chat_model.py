import os
import time
import logging
from abc import ABC, abstractmethod
from app.core.config import LLMConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# These error messages indicate temporary problems that might fix themselves
RETRYABLE_ERROR_MARKERS = ("rate limit", "timeout", "503", "502", "500")


class ChatModel(ABC):
    """The contract every provider implementation must follow."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def _raw_call(self, messages: list[dict]) -> str:
        """Provider-specific call. Each provider implements this differently."""
        raise NotImplementedError

    def chat(self, messages: list[dict]) -> str:
        """Public method every agent calls — retry logic lives here, once, for all providers."""
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._raw_call(messages)
            except Exception as e:
                last_error = e
                is_retryable = any(marker in str(e).lower() for marker in RETRYABLE_ERROR_MARKERS)
                if not is_retryable or attempt == self.config.max_retries:
                    raise
                backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                logger.warning(f"Retryable error on attempt {attempt}: {e}. Retrying in {backoff}s.")
                time.sleep(backoff)
        raise last_error


class GroqChatModel(ChatModel):
    """Groq provider — fast inference on open-source models."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from groq import Groq
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _raw_call(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content


class OpenAIChatModel(ChatModel):
    """OpenAI provider — GPT models."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _raw_call(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content


class OllamaChatModel(ChatModel):
    """Ollama provider — local models, no API key needed."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

    def _raw_call(self, messages: list[dict]) -> str:
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"[Ollama stub] Received: {user_msg}"


def get_chat_model(config: LLMConfig = DEFAULT_CONFIG) -> ChatModel:
    """Factory — the ONLY place that knows which concrete class to build."""
    if config.provider == "groq":
        return GroqChatModel(config)
    elif config.provider == "openai":
        return OpenAIChatModel(config)
    elif config.provider == "ollama":
        return OllamaChatModel(config)
    raise ValueError(f"Unknown provider: {config.provider}")
