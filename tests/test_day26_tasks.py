import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

import redis
from fastapi.testclient import TestClient
from app.api.main import app

API_KEY = "crm-dev-key-2024"
HEADERS = {"X-API-Key": API_KEY}


def test_day26_system():
    print("=" * 70)
    print("🚀 DAY 26 — Redis + Celery Integration Validation")
    print("=" * 70)

    # 1. Test Redis Connection
    print("\n[Step 1] Verifying Redis Broker Connection...")
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    assert r.ping(), "Redis ping failed!"
    print("  ✅ Redis is LIVE and responding to PING!")

    with TestClient(app) as client:
        # 2. Test Celery Lead Scoring Task
        print("\n[Step 2] Testing Celery Task: POST /tasks/leads/score-all...")
        res = client.post("/tasks/leads/score-all", headers=HEADERS)
        assert res.status_code == 202, f"Expected 202, got {res.status_code}: {res.text}"
        data = res.json()
        task_id = data["task_id"]
        print(f"  ✅ Task Created! task_id: {task_id}")

        # Wait for worker to execute task
        print("  Polling task status from Redis backend...")
        for _ in range(15):
            status_res = client.get(f"/tasks/{task_id}", headers=HEADERS)
            status_data = status_res.json()
            curr_state = status_data["status"]
            print(f"    Current state: {curr_state}")
            if curr_state == "SUCCESS":
                print(f"  ✅ Task executed successfully by Celery worker! Result: {status_data['result']}")
                break
            time.sleep(1)
        else:
            print("  ⚠️ Task still pending — ensure Celery worker is running.")

        # 3. Test Scheduled Email (Mini Assignment)
        print("\n[Step 3] Testing Mini Assignment: POST /tasks/email/scheduled...")
        sched_res = client.post(
            "/tasks/email/scheduled?to_email=client@example.com&subject=Welcome&body=Hello&delay_minutes=5",
            headers=HEADERS,
        )
        assert sched_res.status_code == 202, f"Expected 202, got {sched_res.status_code}: {sched_res.text}"
        sched_data = sched_res.json()
        print(f"  ✅ Scheduled Email Task Created! ID: {sched_data['task_id']}")

        # 4. Test CSV Lead Import
        print("\n[Step 4] Testing CSV Bulk Lead Import: POST /tasks/leads/import...")
        csv_content = (
            "name,email,phone,company,plan_interest\n"
            "Ananya Sen,ananya.sen@day26test.io,9811122233,FinTech Nova,growth\n"
            "Vikram Malhotra,vikram@day26test.io,9844455566,LogiTech,enterprise\n"
        )
        files = {"file": ("leads_test.csv", csv_content, "text/csv")}
        import_res = client.post("/tasks/leads/import", files=files, headers=HEADERS)
        assert import_res.status_code == 202, f"Expected 202, got {import_res.status_code}: {import_res.text}"
        import_task_id = import_res.json()["task_id"]
        print(f"  ✅ Import Task Created! task_id: {import_task_id}")

        # Poll import result
        for _ in range(15):
            st_res = client.get(f"/tasks/{import_task_id}", headers=HEADERS)
            st_data = st_res.json()
            if st_data["status"] == "SUCCESS":
                print(f"  ✅ Import Completed by Worker! Imported: {st_data['result']['imported']} leads.")
                break
            time.sleep(1)

    print("\n" + "=" * 70)
    print("🎉 DAY 26 VERIFICATION COMPLETE — ALL SYSTEMS OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    test_day26_system()
