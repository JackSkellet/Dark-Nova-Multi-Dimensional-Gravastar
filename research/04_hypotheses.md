# Hypotheses

## H1: Hierarchical Contextual Routing

Alternative: a contextual hierarchy can reduce active components or memory traffic while preserving quality and latency.

Null: a flat learned router or dense model performs as well after routing overhead, load balance, and route errors are counted.

Initial falsifier: synthetic polysemous token task comparing static token routing, flat contextual routing, hierarchical contextual routing, learned/random controls, route entropy, active components, and latency.

## H2: Multi-Dimensional Or Compositional Weight Representation

Alternative: logical axes such as `Theta[layer, route, component, version]` can be physically encoded with shared storage that improves size-quality-speed tradeoffs.

Null: established quantization or low-rank compression provides a better tradeoff after metadata, caches, and reconstruction workspace are counted.

Initial falsifier: synthetic matrix compression comparing FP32, FP16, int8, int4, low-rank SVD, low-rank plus sparse residual, tensorized blocks, shared bases, and a speculative rank-1 composition.

## H3: Importance-Aware Selective Quantization And Carriers

Alternative: causally important carrier pathways can be identified and protected at higher precision, improving quality per byte over uniform quantization and random protection.

Nulls:

- No sparse set of carrier pathways is consistently important.
- Carrier importance does not transfer across tasks.
- Higher precision than BF16/FP16 gives no material benefit.
- Existing activation-aware quantization captures the useful effect.

Initial falsifier: linear causal-carrier task with activation, gradient, causal ablation, and random rankings.

## H4: Vector Or Indexed Component Lookup

Alternative: vector/index lookup selects large component banks faster or more scalably than dense routing once full lookup and dispatch cost are counted.

Null: ordinary matrix-based routing is faster and equally accurate at realistic local-model scales.

Initial falsifier: CPU component-bank benchmark comparing dense top-k, exact vector search, centroid tree, hash routing, and cached previous route.

## H5: Gated Continual Evolution

Alternative: gated local updates to retrieval, structured memory, adapters, or experts improve future repository tasks while preserving prior tasks and rollback.

Null: frozen model plus continuously updated local retrieval and structured memory performs as well as continual weight updates with lower risk.

Initial falsifier: chronological synthetic API-fact replay comparing frozen memory, retrieval updates, adapter without replay, adapter with replay, and rollback.

