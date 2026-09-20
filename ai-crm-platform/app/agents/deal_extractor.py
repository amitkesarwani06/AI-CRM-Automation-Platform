from langchain_core.output_parsers import PydanticOutputParser
from app.models.deal import DealInfo
from app.prompts.deal_extractor_prompt import deal_extractor_prompt
from app.core.chat_model import get_chat_model


class DealExtractor:
    """Extracts structured deal data from raw text."""

    def __init__(self):
        self.chat_model = get_chat_model()
        self.parser = PydanticOutputParser(pydantic_object=DealInfo)
        self.prompt = deal_extractor_prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def extract(self, raw_text: str) -> DealInfo:
        messages = self.prompt.format_messages(raw_text=raw_text)
        as_dicts = [
            {"role": m.type if m.type != "human" else "user", "content": m.content}
            for m in messages
        ]
        raw_response = self.chat_model.chat(as_dicts)
        return self.parser.parse(raw_response)


if __name__ == "__main__":
    extractor = DealExtractor()

    sample = """
    We're looking at a 5 lakh deal for your Enterprise and WhatsApp modules,
    hopefully closing by end of August.
    """

    deal = extractor.extract(sample)

    print(f"Deal:     {deal.deal_name}")
    print(f"Value:    {deal.value} {deal.currency}")
    print(f"Close:    {deal.expected_close_date}")
    print(f"Products: {deal.product_interest}")
