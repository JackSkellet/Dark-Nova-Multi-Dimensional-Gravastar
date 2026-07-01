# D5 Repository Context Pairwise Probe

Date: 2026-07-01

Result file: `results/D5_repository_context_pairwise_validation.json`

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate_repository_context_pairwise.py --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl --source-split validation --query-split validation --seed 123 --top-k 5 --max-tasks 256 --max-source-rows 10000 --max-query-rows 10000 --min-text-bytes 80 --output results/D5_repository_context_pairwise_validation.json --experiment-id D5_repository_context_pairwise_validation
```

## Hypothesis

Retrieval-augmented repository context should combine with or improve on structured repository memory for API reuse, while keeping hallucinated API suggestions low.

## Setup

This is a reduced D5 validation proxy, not model generation and not executable scoring. The evaluator builds repository-local JavaScript/TypeScript symbol catalogs, creates validation tasks where a query row references same-repository source symbols from other files, and compares:

- `structured_symbol_memory`: known repository source symbols only.
- `retrieved_snippet_identifiers`: identifiers extracted from lexically retrieved source snippets.
- `symbol_aware_retrieved_snippets`: symbols from lexically retrieved source snippets, filtered back to the repository-local extracted source-symbol catalog.
- `query_symbol_aware_retrieval`: repository symbols ranked by explicit query-symbol mention first, then lexical source overlap.

The hallucinated API rate is a proxy: a predicted identifier counts as hallucinated when it is not in the repository-local extracted source-symbol catalog for that task.

## Result

The run builds 23 tasks across 3 validation repositories with 689 extracted candidate symbols.

| Method | Hit@5 | MRR | Coverage@5 | Proxy hallucinated API rate |
| --- | ---: | ---: | ---: | ---: |
| `structured_symbol_memory` | 0.8695652173913043 | 0.753623188405797 | 0.699839291143639 | 0.0 |
| `retrieved_snippet_identifiers` | 0.0 | 0.0 | 0.0 | 0.9826086956521739 |
| `symbol_aware_retrieved_snippets` | 0.21739130434782608 | 0.18478260869565216 | 0.18478260869565216 | 0.0 |
| `query_symbol_aware_retrieval` | 0.782608695652174 | 0.6884057971014493 | 0.6715784215784215 | 0.0 |

## Interpretation

Raw retrieved-snippet identifier extraction is not a viable combination path as implemented. It mostly emits non-API local identifiers and fails the API-reuse proxy.

Symbol-aware retrieved snippets fix the hallucinated-API proxy failure, but still lose badly on recall against structured symbol memory. This points to retrieval ranking as the next failure, not just candidate filtering.

Query-symbol-aware retrieval is a large improvement over lexical symbol-aware retrieval and keeps the proxy hallucinated API rate at 0.0, but it still does not beat structured symbol memory on this D5 validation proxy.

This does not reject retrieval-augmented generation broadly. It rejects raw identifier extraction and keeps `retrieval_augmented_repository_context` in revise until retrieval uses richer repository structure, path/package signals, or a structured-plus-retrieval stack that beats structured memory alone.

## Next Test

Add structured-plus-retrieval context or repository-path-aware tie-breaking. The next variant must beat `structured_symbol_memory` on held-out API reuse before any dense prompt-integration run.
