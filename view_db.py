import sys
import os

# Set terminal encoding for clean display
sys.stdout.reconfigure(encoding='utf-8')

from app.database.connection import SessionLocal, init_db
from app.database.lead_repository import LeadRepository

def show_all_leads():
    init_db()
    with SessionLocal() as session:
        repo = LeadRepository(session)
        leads = repo.get_all(limit=100)
        stats = repo.get_stats()

        print("=" * 85)
        print(" 📊 CRM DATABASE VIEWER (crm_dev.db)")
        print("=" * 85)
        print(f"Total Leads: {stats.get('total_leads', 0)} | "
              f"New: {stats.get('by_status', {}).get('new', 0)} | "
              f"Contacted: {stats.get('by_status', {}).get('contacted', 0)} | "
              f"Qualified: {stats.get('by_status', {}).get('qualified', 0)}")
        print("-" * 85)
        print(f"{'ID':<4} | {'Name':<18} | {'Company':<22} | {'Status':<10} | {'Email':<24}")
        print("-" * 85)

        for lead in leads:
            name = (lead.name or "-")[:18]
            company = (lead.company or "-")[:22]
            status = (lead.status or "-")[:10]
            email = (lead.email or "-")[:24]
            print(f"{lead.id:<4} | {name:<18} | {company:<22} | {status:<10} | {email:<24}")

        print("=" * 85)

if __name__ == "__main__":
    show_all_leads()
