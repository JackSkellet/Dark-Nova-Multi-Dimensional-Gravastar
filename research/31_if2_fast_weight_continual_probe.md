# IF2 Fast-Weight Continual Probe

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_if2_fast_weight_probe.py \
  --output results/IF2_fast_weight_continual_probe.json \
  --experiment-id IF2_fast_weight_continual_probe \
  --seed 123
```

The result was generated from clean commit `eb725fb8e3c16b4a92b4f3a20fa21b93395360d4`.

## Prototype

IF2 tests a signed fast-weight evolution scratchpad as a synthetic continual-learning proxy. It uses four chronological API facts. Each update adds:

- an exact symbol;
- explicit aliases available to structured memory;
- held-out paraphrase queries not stored in updated memory.

Compared methods:

| Method | Description |
| --- | --- |
| exact retrieval | exact updated symbol-to-answer memory |
| structured memory | exact symbol plus explicit alias table |
| fast-weight scratchpad | dense associative feature-hash matrix updated from verified symbols and aliases |
| structured memory plus fast weight | structured memory first, then fast-weight fallback |

## Results

| Metric | Exact retrieval | Structured memory | Fast weight | Structured + fast weight |
| --- | ---: | ---: | ---: | ---: |
| Final accuracy | 0.20 | 0.60 | 0.95 | 0.95 |
| Held-out paraphrases correct | 0 / 8 | 0 / 8 | 7 / 8 | 7 / 8 |
| Storage / parameter bytes | 194 | 596 | 4,096 | 4,692 |

The proxy records `parameter_evolution_adds_value_beyond_updated_memory=true`: the fast-weight path answers held-out paraphrases that exact updated memory and explicit structured aliases do not answer.

## Interpretation

This is a synthetic positive signal for IF2, not model-training evidence. It shows that a parameter-like associative update can add paraphrase generalization beyond updated memory in a controlled fixture, but it pays substantially more storage than structured memory and has no security or poisoning gate yet.

The next IF2 falsifier should run on a real public-repository chronology task with retrieval, structured memory, replay adapter, fast-weight, and combined controls. Promotion to model training requires showing value beyond updated memory on real source/API tasks, not just synthetic paraphrases.

## Source Artifacts

- `results/IF2_fast_weight_continual_probe.json`
- `results/research_assessment.json`
