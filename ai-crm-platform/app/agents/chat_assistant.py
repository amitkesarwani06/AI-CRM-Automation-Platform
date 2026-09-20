from datetime import date
from app.agents.base_agent import BaseAgent
from app.prompts.chat_assistant_prompt import chat_assistant_prompt
from app.memory.memory_manager import get_in_memory_history


class ChatAssistant(BaseAgent):
    """
    Chat Assistant v1 — A conversational CRM agent with multi-turn memory.
    This is the Week 1 Capstone deliverable.
    """

    def __init__(self, user_name: str = "there"):
        super().__init__(
            name="chat_assistant",
            prompt=chat_assistant_prompt,
        )
        self.user_name = user_name
        # Initialize in-memory conversation history
        self.history = get_in_memory_history()

    @property
    def description(self) -> str:
        return "General-purpose CRM chat assistant with conversation memory."

    def run(self, user_input: str, tone_override: str = "") -> str:
        # 1. Pass current conversation history to MessagesPlaceholder("history")
        response = self._call_llm(
            user_input,
            user_name=self.user_name,
            current_date=str(date.today()),
            history=self.history.messages,
            tone_override=tone_override,
        )

        # 2. Save both turns (User + Assistant) to memory for the next turn
        self.history.add_user_message(user_input)
        self.history.add_ai_message(response)

        return response

    def reset_memory(self):
        """Clears all conversation history."""
        self.history.clear()


if __name__ == "__main__":
    agent = ChatAssistant(user_name="Aditya")

    print("=" * 70)
    print("=== Chat Assistant v1 (Interactive Mode with Memory) ===")
    print("Type 'quit' or 'q' to exit | Type 'history' to inspect memory")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "q"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "history":
            print("\n--- Current Conversation History in Memory ---")
            for idx, msg in enumerate(agent.history.messages, 1):
                role = "User" if msg.type == "human" else "Aria"
                print(f"  {idx}. [{role}]: {msg.content}")
            print("-" * 46 + "\n")
            continue

        response = agent.run(user_input)
        print(f"\nAria: {response}\n")
