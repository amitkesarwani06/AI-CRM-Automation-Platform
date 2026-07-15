from app.agents.base_agent import BaseAgent


class GreetingAgent(BaseAgent):
    """A simple greeter that doesn't use the LLM."""

    def __init__(self):
        super().__init__(
            name="greeting_agent",
            system_prompt="Not used",
        )

    @property                          # <-- Implementing the abstract property
    def description(self) -> str:
        return "A simple greeter that welcomes users to the CRM."

    def run(self, user_input: str) -> str:
        return f"Hello! I'm {self.name}, welcome to the CRM!"


if __name__ == "__main__":
    agent = GreetingAgent()
    print(agent)
    print(f"Description: {agent.description}")
    print(agent.run("anything"))
