# Multi-Agent Supervisor Architecture

```mermaid
flowchart TD
    Start([User Message]) --> Supervisor[Supervisor Orchestrator]

    Supervisor -->|sales| Sales[Sales Agent: Maya]
    Supervisor -->|support| Support[Support Agent: Alex]
    Supervisor -->|analytics| Analytics[Analytics Agent: Data Bot]
    Supervisor -->|general| General[General CRM Assistant]

    Sales -->|Single Request Done| Synthesize[Synthesis Node]
    Support -->|Single Request Done| Synthesize
    Analytics -->|Single Request Done| Synthesize
    General -->|Direct Response| Synthesize

    Sales -.->|Multi-Intent Handoff| Analytics
    Support -.->|Multi-Intent Handoff| Analytics

    Synthesize --> End([Final Response to User])

    classDef supervisorStyle fill:#2A60A0,stroke:#1A4070,stroke-width:2px,color:#fff;
    classDef salesStyle fill:#16A085,stroke:#0E6655,stroke-width:2px,color:#fff;
    classDef supportStyle fill:#D35400,stroke:#A04000,stroke-width:2px,color:#fff;
    classDef analyticsStyle fill:#8E44AD,stroke:#6C3483,stroke-width:2px,color:#fff;
    classDef synthStyle fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff;

    class Supervisor supervisorStyle;
    class Sales salesStyle;
    class Support supportStyle;
    class Analytics analyticsStyle;
    class Synthesize synthStyle;
```

---

### How it works:
1. **User Message**: A user query enters the supervisor system.
2. **Supervisor Orchestrator**: Uses LLM classification to decide which specialist(s) should execute the request.
3. **Specialist Agents**:
   - **Sales Agent (Maya)**: Handles lead qualification, pricing, discounts, and follow-up emails.
   - **Support Agent (Alex)**: Performs RAG vector search across product documentation and refund policies.
   - **Analytics Agent (Data Bot)**: Extracts real-time statistics from the SQLite CRM database.
   - **General Assistant**: Directly handles greetings and general capability inquiries.
4. **Agent Handoff**: For complex multi-intent requests (e.g. *"Add lead AND show pipeline"*), the supervisor passes control sequentially between agents.
5. **Synthesis Node**: Merges individual agent outputs into one unified, cohesive response.
