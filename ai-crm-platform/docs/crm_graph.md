# CRM Workflow Graph Architecture

```mermaid
flowchart TD
    Start([Start]) --> Classify[Classify Intent]

    Classify -->|sales| ScoreLead[Score Lead]
    Classify -->|support| Support[Support Node]
    Classify -->|analytics| Analytics[Analytics Node]
    Classify -->|general| General[General Node]

    ScoreLead -->|Info Incomplete| GatherInfo[Gather Missing Info]
    ScoreLead -->|Score 8-10| HotLead[Hot Lead Node]
    ScoreLead -->|Score 5-7| WarmLead[Warm Lead Node]
    ScoreLead -->|Score 1-4| ColdLead[Cold Lead Node]

    HotLead -->|Pending Approval| HITL{{Human-in-the-Loop Gate}}
    HITL -->|Approved| SendEmail[Send Priority Email]

    GatherInfo --> End([End])
    WarmLead --> End
    ColdLead --> End
    SendEmail --> End
    Support --> End
    Analytics --> End
    General --> End

    classDef classifyStyle fill:#2980B9,stroke:#1B4F72,stroke-width:2px,color:#fff;
    classDef hotStyle fill:#C0392B,stroke:#922B21,stroke-width:2px,color:#fff;
    classDef hitlStyle fill:#F39C12,stroke:#B9770E,stroke-width:2px,color:#fff;
    classDef sendStyle fill:#27AE60,stroke:#1E8449,stroke-width:2px,color:#fff;

    class Classify classifyStyle;
    class HotLead hotStyle;
    class HITL hitlStyle;
    class SendEmail sendStyle;
```

---

### Workflow Description:
- **Classify Node**: Detects user intent across sales, support, analytics, and general inquiries.
- **Lead Qualifier**: Evaluates lead completeness and computes an intelligent lead score ($1-10$).
- **Multi-Condition Paths**: Routes to **Hot**, **Warm**, **Cold**, or loops back to request missing information.
- **Human-in-the-Loop (HITL)**: Protects high-value actions by pausing before sending priority emails until human confirmation is granted.
