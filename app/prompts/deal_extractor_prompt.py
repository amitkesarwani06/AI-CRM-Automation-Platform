from langchain_core.prompts import ChatPromptTemplate

DEAL_EXTRACTOR_SYSTEM = """\
You are a CRM deal extraction specialist.

Your job is to read a raw message and extract deal/opportunity information from it.

RULES:
1. Only extract information explicitly stated — never invent values.
2. If a field is not mentioned, omit it (it will default to null).
3. Respond ONLY with the JSON object described below — no explanation.

{format_instructions}
"""

deal_extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", DEAL_EXTRACTOR_SYSTEM),
    ("human", "Extract deal data from this message:\n\n{raw_text}"),
])
