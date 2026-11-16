import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Gemini client initialization
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def ask_llm(question: str, system_prompt: str = "You are a helpful CRM assistant.") -> str:
    """The atomic unit of every AI feature we'll build."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    print(ask_llm("Say hello to a new CRM user named Aditya."))
