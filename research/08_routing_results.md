# Routing Results

Status: generated from `results/E1_contextual_routing.json` and `results/E1b_routing_robustness.json`.

Command:

```bash
uv run python scripts/run_experiments.py --seed 123
```

Initial interpretation rules:

- Static token routing should fail on polysemous tokens.
- Contextual routing must beat random routing.
- Hierarchical routing is only promising if it preserves quality while reducing active components or measured traffic.
- Synthetic route labels do not prove semantic specialization.

## Seed 123 Results

| Method | Route accuracy | Active components | Estimated traffic units | p95 routing latency ms | Max expert share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static token | 0.502 | 2 | 2000 | 0.053 | 0.508 |
| Flat contextual | 1.000 | 4 | 4000 | 0.113 | 0.263 |
| Hierarchical contextual | 1.000 | 3 | 3000 | 0.399 | 0.263 |
| Random control | 0.258 | 4 | 4000 | 0.865 | 0.256 |

## Interpretation

The reduced experiment supports the correction that isolated token IDs are insufficient: static token routing fails on polysemy. The hierarchy preserved quality and reduced the toy active-component count by 25% versus flat contextual routing, but its p95 router latency was higher. This is not yet a Pareto improvement because the benchmark is synthetic and does not include real model dispatch or memory traffic.

## E1b: Mixed And Adversarial Routing Robustness

Seed 123 follow-up with single-route, mixed-route, and adversarial-cue contexts:

| Method | Exact set accuracy | Route recall | Mixed-route recall | Adversarial exact accuracy | False positive routes/sample | Active components | p95 routing latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-label contextual | 0.667 | 0.833 | 0.500 | 1.000 | 0.000 | 1 | 0.138 |
| Multi-label contextual | 0.742 | 1.000 | 1.000 | 0.735 | 0.258 | 2 | 0.563 |
| Latent-centroid multi-label | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2 | 2.779 |

Interpretation: single-label contextual routing is inadequate for mixed-topic prompts because it can only select one route. Naive multi-label routing recovers all true routes but adds false positive routes under overlapping/adversarial cues. The latent-centroid multi-label router solves this synthetic robustness set, but at much higher routing latency than the keyword routers. This is useful evidence for multi-label routing as a requirement, not evidence that a hierarchy is faster or semantically causal in a real model.
