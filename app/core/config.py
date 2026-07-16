from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """Centralizes all LLM settings — change once, applies everywhere."""
    provider: str = os.getenv("LLM_PROVIDER", "groq")        # "groq" or "openai"
    model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    temperature: float = 0.3
    max_retries: int = 3
    timeout_seconds: int = 30
    max_tokens: int = 256


DEFAULT_CONFIG = LLMConfig()
