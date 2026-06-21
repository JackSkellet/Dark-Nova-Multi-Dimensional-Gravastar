# Idea Foundry Candidate Lane

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_idea_foundry.py \
  --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl \
  --candidates-output results/idea_foundry_candidates.json \
  --probe-output results/IF1_repository_graph_signal_probe.json \
  --experiment-id idea_foundry \
  --seed 123
```

The command records six architecture candidates and runs the first cheap falsifying probe for IF1. The records were generated from clean commit `38f7e3f33c34a8be8672637d3984db0e7a1e8da2`.

## Candidate Set

| ID | Name | Family | Uses adapters | Uses MoE/topic routing | Novelty label | Primary target |
| --- | --- | --- | --- | --- | --- | --- |
| IF1 | Repository graph conditioned decoder | code structure | no | no | adjacent | repository graph/test/doc structure |
| IF2 | Signed fast-weight evolution scratchpad | continual evolution | no | no | adjacent | local chronological updates |
| IF3 | Audited block-codebook weight generator | compression | no | no | potentially novel | metadata-counted weight storage |
| IF4 | Executable trace contrastive objective | code-document objective | no | no | adjacent | tests/docs/source alignment |
| IF5 | Syntax-state recurrent mixer | state-space hybrid | no | no | adjacent | code syntax state and long spans |
| IF6 | Delta-consolidated specialist lattice | continual specialization | yes | yes | adjacent | repository specialists plus consolidation |

Constraint summary:

- Candidate count: 6.
- Without adapters: 5.
- Without MoE/topic routing: 5.
- Continual-evolution candidates: 2.
- Compression candidates: 1.
- Code-structure candidates: 3.
- Potentially novel candidates: 1.

Each candidate record includes mechanism/equations, closest primary-source prior art, exact difference, expected scaling, ROCm plan, likely failure, cheapest falsifying test, and mechanism occurrence evidence.

## IF1 Probe

`IF1_repository_graph_signal_probe` scanned all 27,915 D5 rows with regex import/require/export extraction and repository-local relative path resolution.

| Metric | Value |
| --- | ---: |
| Documents scanned | 27,915 |
| Regex import/export/require edges | 18,919 |
| Locally resolved relative edges | 12 |
| Repositories with edges | 4,041 |
| Repository-aware splits preserved | true |
| Mechanism signal present | true |

Role counts in the scanned D5 rows:

| Role | Rows |
| --- | ---: |
| source | 24,079 |
| docstring | 8,655 |
| test | 3,958 |
| documentation | 1,157 |
| readme | 351 |
| changelog | 21 |

## Interpretation

IF1 has enough D5 structure to justify a better graph extractor, but not enough resolved local edges from the current regex-only probe to justify a 5-10M-token graph-conditioned model pilot yet. The key limitation is not absence of signal; it is weak extraction and package resolution.

The next smallest IF1 prototype should add AST/package-aware JavaScript import resolution and same-repository symbol/test/doc linking, then rerun this probe. A model pilot should require materially more resolved local edges and a fixed edge-bias construction that preserves repository-aware splits.

## Source Artifacts

- `results/idea_foundry_candidates.json`
- `results/IF1_repository_graph_signal_probe.json`
- `results/research_assessment.json`
