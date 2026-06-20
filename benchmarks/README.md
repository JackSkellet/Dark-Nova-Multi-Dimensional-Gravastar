# Benchmarks

Initial benchmarks are run through:

```bash
uv run python scripts/run_experiments.py --seed 123
```

The first pass measures CPU latency for small synthetic routing, lookup, compression, and continual-memory operations. Larger benchmarks must document hardware, batch size, context length, model/data provenance, and every storage and cache overhead.

For the explicit public-repository chronology benchmark:

```bash
git clone https://github.com/antirez/kilo data/raw/public/kilo
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/kilo --max-commits 12 --top-k 5 --include-symbol-qa --include-stale-docs
```

This reads a local clone and writes `results/E5_public_repo_chronology.json`, `results/E5_public_repo_symbol_qa.json`, and `results/E5_public_repo_stale_docs.json`.

For the larger public sample:

```bash
git clone https://github.com/pallets/markupsafe data/raw/public/markupsafe
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/markupsafe --max-commits 30 --top-k 5 --include-symbol-qa --include-stale-docs --output results/E5_markupsafe_chronology.json --symbol-output results/E5_markupsafe_symbol_qa.json --stale-doc-output results/E5_markupsafe_stale_docs.json
```
