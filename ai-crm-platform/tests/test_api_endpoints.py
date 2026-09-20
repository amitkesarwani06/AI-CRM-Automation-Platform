import os
import sys

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from app.api.main import app


API_KEY = "crm-dev-key-2024"
HEADERS = {"X-API-Key": API_KEY}


def run_api_tests():
    print("=" * 70)
    print("🚀 DAY 25 — FastAPI REST API Test Suite")
    print("=" * 70)

    with TestClient(app) as client:
        # ── Test 1: Root & Health Check ───────────────────────────────────────
        print("\n[Test 1] Testing Root and Health Check Endpoints...")
        r_root = client.get("/")
        assert r_root.status_code == 200, f"Root failed: {r_root.text}"
        print(f"  GET / -> {r_root.status_code} | {r_root.json()['service']}")

        r_health = client.get("/health")
        assert r_health.status_code == 200, f"Health failed: {r_health.text}"
        data = r_health.json()
        print(f"  GET /health -> {r_health.status_code} | DB: {data['database']} | Model: {data['ai_model']}")
        assert data["database"] == "connected", "Database should be connected"

        # ── Test 2: Authentication Enforcement ────────────────────────────────
        print("\n[Test 2] Testing API Key Authentication...")
        r_unauth = client.get("/leads/")
        print(f"  GET /leads without header -> {r_unauth.status_code} (Expected 401)")
        assert r_unauth.status_code == 401, f"Expected 401, got {r_unauth.status_code}"

        r_bad_key = client.get("/leads/", headers={"X-API-Key": "wrong-key"})
        print(f"  GET /leads with invalid key -> {r_bad_key.status_code} (Expected 401)")
        assert r_bad_key.status_code == 401, f"Expected 401, got {r_bad_key.status_code}"

        r_auth = client.get("/leads/", headers=HEADERS)
        print(f"  GET /leads with valid key -> {r_auth.status_code} (Expected 200)")
        assert r_auth.status_code == 200, f"Expected 200, got {r_auth.status_code}"

        # ── Test 3: Lead Management CRUD ──────────────────────────────────────
        print("\n[Test 3] Testing Lead CRUD Endpoints...")
        new_lead_payload = {
            "name": "Priya Sharma",
            "email": "priya.sharma@startupxyz.com",
            "phone": "+91 9876543210",
            "company": "StartupXYZ",
            "city": "Mumbai",
            "plan_interest": "growth",
            "notes": "Testing Day 25 FastAPI integration",
        }
        r_create = client.post("/leads/", json=new_lead_payload, headers=HEADERS)
        assert r_create.status_code == 201, f"Create lead failed: {r_create.text}"
        created_lead = r_create.json()
        lead_id = created_lead["id"]
        print(f"  POST /leads/ -> Created Lead #{lead_id} ({created_lead['name']}, Company: {created_lead['company']})")

        # Get by ID
        r_get = client.get(f"/leads/{lead_id}", headers=HEADERS)
        assert r_get.status_code == 200
        print(f"  GET /leads/{lead_id} -> Found {r_get.json()['name']}")

        # Patch Lead
        patch_payload = {"status": "contacted", "score": 8, "notes": "Contacted via phone demo"}
        r_patch = client.patch(f"/leads/{lead_id}", json=patch_payload, headers=HEADERS)
        assert r_patch.status_code == 200
        updated = r_patch.json()
        print(f"  PATCH /leads/{lead_id} -> Status: {updated['status']}, Score: {updated['score']}")
        assert updated["status"] == "contacted"
        assert updated["score"] == 8

        # ── Test 4: Pipeline Statistics Overview ──────────────────────────────
        print("\n[Test 4] Testing Pipeline Stats Overview...")
        r_stats = client.get("/leads/stats/overview", headers=HEADERS)
        assert r_stats.status_code == 200
        stats = r_stats.json()
        print(f"  GET /leads/stats/overview -> Total leads: {stats['total_leads']}, Breakdown: {stats['by_status']}")

        # ── Test 5: Mini Assignment — AI Lead Extraction ──────────────────────
        print("\n[Test 5] Testing AI Lead Extraction (Mini Assignment)...")
        extract_payload = {
            "text": "Hey, I'm Rahul Verma from TechVibe Pune. We need a CRM for our 20 agents. Reach me at rahul.verma@techvibe.io or 9823001122. We are very eager to start immediately!"
        }
        r_extract = client.post("/leads/extract", json=extract_payload, headers=HEADERS)
        if r_extract.status_code == 201:
            extracted = r_extract.json()
            extracted_id = extracted["id"]
            print(f"  POST /leads/extract -> Extracted #{extracted_id} {extracted['name']} | Company: {extracted['company']} | Score: {extracted['score']}")
            # Cleanup extracted test lead
            client.delete(f"/leads/{extracted_id}", headers=HEADERS)
        else:
            print(f"  POST /leads/extract returned status {r_extract.status_code}: {r_extract.text}")

        # Delete created test lead
        r_del = client.delete(f"/leads/{lead_id}", headers=HEADERS)
        assert r_del.status_code == 204
        print(f"  DELETE /leads/{lead_id} -> 204 No Content (Deleted test lead)")

        # ── Test 6: Knowledge Base RAG Direct Q&A ─────────────────────────────
        print("\n[Test 6] Testing Knowledge Base RAG Direct Q&A...")
        rag_payload = {"message": "What is your refund policy and pricing?"}
        r_rag = client.post("/chat/rag", json=rag_payload, headers=HEADERS)
        assert r_rag.status_code == 200, f"RAG failed: {r_rag.text}"
        rag_res = r_rag.json()
        print(f"  POST /chat/rag -> Agents: {rag_res['agents_used']}")
        print(f"  Response: {rag_res['response'][:100]}...")

        # ── Test 7: Multi-Agent Chat Supervisor ───────────────────────────────
        print("\n[Test 7] Testing Multi-Agent Supervisor Chat...")
        chat_payload = {"message": "Hello! What can this CRM automation platform do?"}
        r_chat = client.post("/chat/", json=chat_payload, headers=HEADERS)
        assert r_chat.status_code == 200, f"Chat failed: {r_chat.text}"
        chat_res = r_chat.json()
        print(f"  POST /chat/ -> Intent: {chat_res['intent']} | Agents: {chat_res['agents_used']}")
        print(f"  Response: {chat_res['response'][:120]}...")

    print("\n" + "=" * 70)
    print("✅ ALL DAY 25 FASTAPI BACKEND TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_api_tests()
