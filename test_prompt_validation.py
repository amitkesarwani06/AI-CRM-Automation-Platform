from app.prompts.chat_assistant_prompt import chat_assistant_prompt

print("=" * 60)
print("TESTING PROMPT TEMPLATE VALIDATION")
print("=" * 60)

try:
    # Intentionally missing 'current_date' and 'tone_override'
    print("Attempting to format template with missing variables...")
    messages = chat_assistant_prompt.format_messages(
        input="Hi!",
        user_name="Aditya"
    )
except KeyError as e:
    print(f"\n✅ Successfully caught expected KeyError!")
    print(f"Missing required variable: {e}")
    print("\nThis is a good thing — LangChain caught our missing variable before sending a broken prompt to the LLM!")

print("-" * 60)
