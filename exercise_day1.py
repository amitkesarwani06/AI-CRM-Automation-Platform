from app.core.llm_client import ask_llm

question = "How can I help a customer who wants to cancel their subscription?"

formal = ask_llm(question, system_prompt="You are a formal, professional CRM assistant. Use business language.")
casual = ask_llm(question, system_prompt="You are a friendly, casual CRM buddy. Use simple, warm language.")

print("=" * 60)
print("FORMAL RESPONSE:")
print("=" * 60)
print(formal)
print()
print("=" * 60)
print("CASUAL RESPONSE:")
print("=" * 60)
print(casual)
