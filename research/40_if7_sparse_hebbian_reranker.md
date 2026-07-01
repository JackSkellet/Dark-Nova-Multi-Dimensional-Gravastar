# IF7 Sparse Hebbian Candidate Reranker

Date: 2026-07-01

## Scope

This follow-up tests a different IF7 integration after dense cue-plus-Hebbian
concatenation failed. Instead of feeding the full 2,048-node Hebbian score vector
into a trained model, it uses the Hebbian memory only to produce a sparse top-k
candidate set, then trains a tiny candidate scorer.

Two 500k-window runs were recorded:

- `IF7e_sparse_hebbian_reranker_d5_500k_windows`: sparse candidate features from
  Hebbian score, rank, frequency, and cue membership.
- `IF7f_sparse_hebbian_reranker_priors_d5_500k_windows`: adds train-derived node
  cue/target priors.

Both use 494,403 D5 train window patterns, 24,216 validation window patterns, 64
Hebbian candidates per pattern, and ROCm-backed Torch training.

## Results

At recall@32:

| Method | Hit@32 | Coverage@32 | MRR |
| --- | ---: | ---: | ---: |
| Raw Hebbian memory | 0.8167327386851668 | 0.10158472099068659 | 0.193942667285143 |
| Candidate ceiling | 0.9252560290716881 | 0.17033596353486855 | 0.9252560290716881 |
| IF7e sparse reranker | 0.7385612817971589 | 0.07207645355136083 | 0.16757045744935523 |
| IF7f sparse reranker plus priors | 0.7951354476379253 | 0.08836340463817875 | 0.18669151337014772 |

The prior features improve the trained reranker over IF7e, but the reranker still
does not beat raw Hebbian ranking. The candidate ceiling is much higher than raw
Hebbian hit rate, so there is useful candidate-set headroom, but the current tiny
scorer does not exploit it.

## Interpretation

The IF7 evidence now separates into three pieces:

- Sparse Hebbian co-activation has real associative signal.
- Dense full-score Hebbian conditioning hurts trained validation quality.
- Sparse top-k candidate reranking has headroom but the current scorer is still
  weaker than raw Hebbian ordering.

The next IF7 work should not add another tiny linear scorer. It should either use
the Hebbian candidate set as retrieval context for a stronger model, or change the
task to repository-level relevant-file/source-doc/test linking where candidate
identity and file metadata can be scored directly.

## Source Artifacts

- `src/weightlab/if7_sparse_hebbian.py`
- `scripts/train_if7_sparse_reranker.py`
- `results/IF7e_sparse_hebbian_reranker_d5_500k_windows.json`
- `results/IF7f_sparse_hebbian_reranker_priors_d5_500k_windows.json`
