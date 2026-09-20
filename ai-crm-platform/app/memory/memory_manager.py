from langchain_core.chat_history import InMemoryChatMessageHistory


def get_in_memory_history() -> InMemoryChatMessageHistory:
    """Returns an in-memory chat history store for tracking conversation turns."""
    return InMemoryChatMessageHistory()
