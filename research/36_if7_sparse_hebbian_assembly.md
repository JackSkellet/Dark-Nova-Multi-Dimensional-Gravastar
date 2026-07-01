# IF7 Sparse Hebbian Assembly Memory

Date: 2026-07-01

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_if7_sparse_hebbian_probe.py \
  --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl \
  --split train \
  --node-count 4096 \
  --max-active-nodes 48 \
  --max-train-rows 8192 \
  --max-eval-rows 512 \
  --recall-at-k 32 \
  --output results/IF7_sparse_hebbian_d5_probe.json \
  --experiment-id IF7_sparse_hebbian_d5_probe \
  --seed 123
```

## Scope

IF7 tests a sparse brain-inspired associative memory: each real corpus row activates
only a small assembly of hashed feature nodes, and nodes that co-activate strengthen
pairwise bonds. Evaluation masks each exposed row down to cue nodes from repository,
path, language, and role metadata, then asks whether the learned assembly can recover
withheld text-derived identifier/import nodes.

This is an associative-memory mechanism probe on real D5 corpus rows. It is not
language-model training, executable coding evaluation, unseen-repository
generalization, or a ROCm kernel benchmark.

## Result

The run sampled 8,192 D5 train rows from 26,173 scanned train rows. It produced
8,190 usable sparse patterns from 7,381 repositories and evaluated 512 masked
patterns. Mean active nodes were 46.08 out of 4,096, so the mean active fraction was
0.01125.

At recall@32:

| Method | Hit@32 | Coverage@32 | MRR |
| --- | ---: | ---: | ---: |
| Frequency control | 0.884765625 | 0.10871456122872884 | 0.2834511109954672 |
| Random sparse control | 0.201171875 | 0.007418960394822318 | 0.027356396479057583 |
| Hebbian sparse assembly | 0.953125 | 0.16677796920591573 | 0.47986234163805885 |

The Hebbian assembly therefore beats both frequency and random sparse controls on
hit rate, target coverage, and reciprocal rank for this masked real-row recall task.
The record reports `sparse_hebbian_adds_signal_over_frequency_and_random_controls`.

Storage accounting for the current NumPy probe is not optimized: the Hebbian matrix
plus label index accounts for 69,399,884 bytes, while the random sparse control
accounts for 67,141,632 bytes. This is acceptable for a mechanism probe but not a
deployment representation.

## Interpretation

IF7 has a real D5 mechanism signal: sparse co-activation bonds recover withheld
identifier/import nodes better than global node frequency and random sparse
assemblies while activating about 1.1 percent of the node space per row.

The result is not yet a model-quality or architecture-frontier claim. The current
task evaluates masked recall from rows that were already exposed to the assembly.
That is the correct first test for "nodes that fire together wire together", but it
does not prove that the memory improves code generation, repository reasoning,
repair, documentation quality, or unseen-repository generalization.

## Next Step

Promote IF7 only to a stronger benchmark, not directly into the decoder. The next
test should use repository-heldout association tasks or the repository-balanced
benchmark scaffold: source-to-test/doc linking, relevant-file selection, API-reuse
cue completion, or documentation/source consistency. If IF7 adds value there, the
next architecture experiment is a retrieval-plus-Hebbian conditioning path for the
dense-528 or D5 BPE pilot.

## Source Artifacts

- `src/weightlab/if7_sparse_hebbian.py`
- `scripts/run_if7_sparse_hebbian_probe.py`
- `tests/test_if7_sparse_hebbian.py`
- `results/IF7_sparse_hebbian_d5_probe.json`
