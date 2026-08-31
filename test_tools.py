import os
import sys
sys.path.insert(0, os.getcwd())

from app.tools.lead_tools import create_lead, get_lead, update_lead
from app.tools.utility_tools import get_current_datetime, calculate_discount
from app.tools.knowledge_base_tool import get_pricing

print("\n" + "="*60)
print("DAY 15 — Custom Tools Tests")
print("Week 3 Begins — Agents that ACT!")
print("="*60)

# TEST 1: Tool Schema
print("\n" + "="*60)
print("TEST 1: Tool Schema — what the agent sees")
print("="*60)
for t in [create_lead, get_lead, get_pricing, get_current_datetime]:
    print(f"\nTool: {t.name}")
    print(f"Description: {t.description[:80].strip()}...")
    print(f"Args: {list(t.args.keys())}")

# TEST 2: create_lead
print("\n" + "="*60)
print("TEST 2: create_lead")
print("="*60)
result = create_lead.invoke({
    "name": "Rahul Sharma",
    "email": "rahul@techcorp.com",
    "phone": "9823456710",
    "company": "TechCorp Solutions",
    "plan_interest": "growth",
    "notes": "Spoke at conference, very interested in AI features",
})
print(result)

result2 = create_lead.invoke({
    "name": "Priya Mehta",
    "company": "StartupXYZ",
    "plan_interest": "starter",
})
print("\n" + result2)

# TEST 3: get_lead
print("\n" + "="*60)
print("TEST 3: get_lead")
print("="*60)
print("By name:")
print(get_lead.invoke({"name": "Rahul"}))
print("\nBy ID:")
print(get_lead.invoke({"lead_id": 1}))
print("\nAll leads:")
print(get_lead.invoke({}))

# TEST 4: update_lead
print("\n" + "="*60)
print("TEST 4: update_lead")
print("="*60)
print(update_lead.invoke({
    "lead_id": 1,
    "status": "contacted",
    "notes": "Called Monday, scheduled demo for Friday",
}))

# TEST 5: get_pricing
print("\n" + "="*60)
print("TEST 5: get_pricing")
print("="*60)
print("All plans:")
print(get_pricing.invoke({"plan_name": "all"}))
print("\nGrowth only:")
print(get_pricing.invoke({"plan_name": "growth"}))

# TEST 6: Utility tools
print("\n" + "="*60)
print("TEST 6: Utility tools")
print("="*60)
print("Current datetime:")
print(get_current_datetime.invoke({}))
print("\nDiscount calculation:")
print(calculate_discount.invoke({"original_price": 2999, "discount_percent": 20}))

# TEST 7: ALL_CRM_TOOLS
print("\n" + "="*60)
print("TEST 7: ALL_CRM_TOOLS list")
print("="*60)
from app.tools import ALL_CRM_TOOLS
print(f"Total tools: {len(ALL_CRM_TOOLS)}\n")
for i, t in enumerate(ALL_CRM_TOOLS, 1):
    print(f"{i}. {t.name}")

print("\n" + "="*60)
print("Day 15 COMPLETE — Custom Tools working!")
print("Next -> Day 16: Tool Binding + Tool Calling")
print("="*60)
