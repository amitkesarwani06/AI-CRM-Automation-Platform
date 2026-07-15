import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq client initialization
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(user_message: str, system_prompt: str = "You are a helpful CRM assistant.") -> str:
    """The atomic unit of every AI feature we'll build."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask_llm("Say hello to a new CRM user named Aditya."))
