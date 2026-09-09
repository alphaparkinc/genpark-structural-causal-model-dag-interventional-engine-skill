# GenPark Structural Causal Model DAG Interventional Engine Skill

Structural Causal Model (SCM) DAG engine evaluating observational distributions and simulating Pearl's do-calculus interventions.

Read more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    S[Seasonality] -->|Causal| M[MarketingSpend]
    S -->|Direct Confounder| T[WebTraffic]
    M -->|Causal| T
    T -->|Causal| R[Revenue]
    style S fill:#e1f5fe
    style M fill:#fff9c4
    style T fill:#c8e6c9
    style R fill:#d1c4e9
```

## Features
- Pure Python standard library causal DAG representation.
- Pearl's do(X = x) graph surgery removing incoming directed edges.
- Exact structural state propagation across endogenous networks.
