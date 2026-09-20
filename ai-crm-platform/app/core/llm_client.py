import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq client initialization
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(question: str, system_prompt: str = "You are a helpful CRM assistant.") -> str:
    """The atomic unit of every AI feature we'll build."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask_llm("Say hello to a new CRM user named Aditya."))
