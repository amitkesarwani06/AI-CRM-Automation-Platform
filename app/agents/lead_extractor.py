from langchain_core.output_parsers import PydanticOutputParser
from app.models.lead import LeadInfo
from app.prompts.lead_extractor_prompt import lead_extractor_prompt
from app.core.chat_model import get_chat_model
from langchain_core.exceptions import OutputParserException



class LeadExtractor:
    """
    Reads raw text (email/chat/transcript) and returns a validated LeadInfo object.

    This is NOT a BaseAgent subclass — it's a simple input→output extractor.
    Not everything needs to be a conversational agent.
    """

    def __init__(self):
        self.chat_model = get_chat_model()
        self.parser = PydanticOutputParser(pydantic_object=LeadInfo)

        # Pre-fill format_instructions into the prompt template
        # This tells the LLM exactly what JSON schema to produce
        self.prompt = lead_extractor_prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def extract(self, raw_text: str) -> LeadInfo | None:
        try:
            # 1. Format the prompt with the raw text
            messages = self.prompt.format_messages(raw_text=raw_text)

            # 2. Convert LangChain messages to plain dicts for our ChatModel
            as_dicts = [
                {"role": m.type if m.type != "human" else "user", "content": m.content}
                for m in messages
            ]

            # 3. Call the LLM
            raw_response = self.chat_model.chat(as_dicts)

            # 4. Parse the raw JSON string into a validated LeadInfo object
            return self.parser.parse(raw_response)
        except OutputParserException as e:
            print(f"Parse failed: {e}")
            return None

if __name__ == "__main__":
    extractor = LeadExtractor()

    # Test 1: Normal extraction
    sample = """
    Hi, my name is Priya Sharma. I'm the Head of Sales at MegaTech Solutions
    in Pune. We have a team of 50 salespeople and we're urgently looking for a
    CRM that can automate our lead follow-ups and email sequences.
    Please reach me at priya.sharma@megatech.com or call 9823456710.
    We'd like a demo this week if possible.
    """

    print("--- Test 1: Normal extraction ---")
    lead = extractor.extract(sample)
    if lead:
        print(f"Name:    {lead.name}")
        print(f"Email:   {lead.email}")
        print(f"Phone:   {lead.phone}")
        print(f"Company: {lead.company}")
        print(f"Score:   {lead.score}/10")

    # Test 2: Deliberately bad input
    print("\n--- Test 2: Garbage input ---")
    bad_result = extractor.extract("lol just random gibberish with no lead info 12345")
    print(f"Result: {bad_result}")
