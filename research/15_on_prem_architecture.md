# On-Premises Architecture

```mermaid
flowchart LR
  U["User request"] --> A["External authorization service"]
  A --> C["Context builder"]
  C --> R["Local retrieval and structured repo memory"]
  C --> M["Local model runtime"]
  R --> M
  M --> V["Validation tools"]
  V --> O["Response or patch"]
  D["Approved artifacts"] --> G["Offline update controller"]
  G --> E["Evaluation and security gates"]
  E --> S["Signed adapters/indexes"]
  S --> M
```

Production must operate without cloud inference, cloud embeddings, external telemetry, external vector databases, external training services, unapproved network access, or uploading prompts/code/indexes/logs/model updates.

Network access in this research repository is limited to explicit literature/source retrieval and package installation during public research.

