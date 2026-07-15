from app.agents.base_agent import BaseAgent


class ChatAssistant(BaseAgent):
    """A friendly CRM chat assistant — the first concrete agent."""

    def __init__(self):
        super().__init__(
            name="chat_assistant",
            system_prompt="You are a friendly, concise CRM chat assistant.",
        )

    @property                          # <-- Implementing the abstract property
    def description(self) -> str:
        return "A friendly assistant that answers general CRM questions."

    def run(self, user_input: str) -> str:
        return self._call_llm(user_input)


if __name__ == "__main__":
    agent = ChatAssistant()
    print(agent)
    print(f"Description: {agent.description}")   # <-- No () needed, it's a property
    print(agent.run("What can you help me with?"))
