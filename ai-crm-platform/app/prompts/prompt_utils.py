def with_examples(base_prompt: str, examples: list[tuple[str, str]]) -> str:
    """Appends few-shot examples to a base prompt in a consistent format.

    Args:
        base_prompt: The system prompt text.
        examples: A list of (user_message, ideal_response) tuples.

    Returns:
        The prompt with examples appended.
    """
    if not examples:
        return base_prompt
    formatted = "\n\nEXAMPLES:\n"
    for user_msg, ideal_response in examples:
        formatted += f"User: {user_msg}\nAssistant: {ideal_response}\n\n"
    return base_prompt + formatted
