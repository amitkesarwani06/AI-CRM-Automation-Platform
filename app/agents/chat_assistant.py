from datetime import date
from app.agents.base_agent import BaseAgent
from app.prompts.chat_assistant_prompt import chat_assistant_prompt


class ChatAssistant(BaseAgent):
    """A friendly CRM chat assistant — the first concrete agent."""

    def __init__(self):
        super().__init__(
            name="chat_assistant",
            prompt=chat_assistant_prompt,
        )

    @property
    def description(self) -> str:
        return "A friendly assistant that answers general CRM questions."

    def run(self, user_input: str, user_name: str = "there", tone_override: str = "") -> str:
        # Injects current user name, date, and tone override into the template at runtime
        return self._call_llm(
            user_input,
            user_name=user_name,
            current_date=str(date.today()),
            tone_override=tone_override,
        )


if __name__ == "__main__":
    agent = ChatAssistant()
    print(agent)
    print(f"Description: {agent.description}")
    
    print("\n--- Testing default tone ---")
    print(agent.run("What can you help me with?", user_name="Aditya"))
    
    print("\n--- Testing with tone override (Enthusiastic + Emojis) ---")
    print(agent.run(
        "What can you help me with?", 
        user_name="Aditya", 
        tone_override="Tone override: Be extremely enthusiastic and use lots of emojis!"
    ))
