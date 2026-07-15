from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """Centralizes all LLM settings — change once, applies everywhere."""
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.3


DEFAULT_CONFIG = LLMConfig()
