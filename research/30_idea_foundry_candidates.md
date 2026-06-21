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

The command records six architecture candidates and runs the first cheap falsifying probe for IF1. The current records were generated from clean commit `79dd6679833cfda0ece6d44fe0ea686c48149eec`.

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

`IF1_repository_graph_signal_probe` scanned all 27,915 D5 rows with regex import/require/export extraction, repository-local relative path resolution, and heuristic test/doc role links. Docstring/source self-links are excluded.

| Metric | Value |
| --- | ---: |
| Documents scanned | 27,915 |
| Regex import/export/require edges | 18,919 |
| Locally resolved import edges | 12 |
| Heuristic role-link edges | 313 |
| Total graph edges | 325 |
| Doc-to-source edges | 312 |
| Test-to-source edges | 1 |
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

IF1 has enough D5 structure to justify continued graph-extraction work, but the edge mix is not yet strong enough for a 5-10M-token graph-conditioned model pilot. Import resolution remains weak at 12 resolved local import edges. The added role links raise usable graph edges to 325, but 312 are heuristic doc-to-source links and only one is test-to-source. The key limitation is still weak extraction and package resolution, not absence of repository metadata.

The next smallest IF1 prototype should add AST/package-aware JavaScript import resolution and stricter same-repository symbol/test/doc linking, then rerun this probe. A model pilot should require materially more non-heuristic resolved edges and a fixed edge-bias construction that preserves repository-aware splits.

## Source Artifacts

- `results/idea_foundry_candidates.json`
- `results/IF1_repository_graph_signal_probe.json`
- `results/research_assessment.json`
