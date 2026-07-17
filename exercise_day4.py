from app.core.chat_model import get_chat_model

# 1. Define prompts
VAGUE_PROMPT = "You are a lead qualification assistant. Ask the user questions to qualify them as a lead."

ENGINEERED_PROMPT = """\
You are a lead qualification assistant. Your goal is to qualify lead details by gathering specific information.

REQUIRED DETAILS TO GATHER:
1. Company Size (number of employees)
2. Budget (monthly or annual spend capability)
3. Main Pain Point or Goal

CONSTRAINTS:
- Ask only ONE question at a time to keep it conversationally friendly.
- Do NOT offer product details or pricing yet.
- Once all three details are collected, output "QUALIFICATION COMPLETE: [summary]" and stop asking questions.
- If the user asks something irrelevant, politely redirect them back to the qualification flow.
"""

# 2. Set up test inputs
test_inputs = [
    "Hi, I'm interested in buying your software.",
    "I'm looking for a tool for our 50-person team, budget is around $2000/mo.",
    "Can you tell me a joke?"
]

# 3. Initialize model
model = get_chat_model()

print("=" * 80)
print("RUNNING PROMPT COMPARISON")
print("=" * 80)

for i, user_input in enumerate(test_inputs, 1):
    print(f"\n--- TEST CASE {i}: \"{user_input}\" ---")
    
    # Run vague prompt
    vague_response = model.chat([
        {"role": "system", "content": VAGUE_PROMPT},
        {"role": "user", "content": user_input}
    ])
    
    # Run engineered prompt
    engineered_response = model.chat([
        {"role": "system", "content": ENGINEERED_PROMPT},
        {"role": "user", "content": user_input}
    ])
    
    print("\n[VAGUE PROMPT RESPONSE]:")
    print(vague_response)
    print("\n[ENGINEERED PROMPT RESPONSE]:")
    print(engineered_response)
    print("-" * 50)
