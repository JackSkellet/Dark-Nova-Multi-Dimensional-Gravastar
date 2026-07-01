# IF7 Hebbian Trained Model 2M-Window Run

Date: 2026-07-01

## Hypothesis

If sparse Hebbian co-activation memory captures useful "nodes that fire together wire together" structure, then a trained cue-plus-Hebbian model should outperform a cue-only trained model on a much larger D5 window-pattern task.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_if7_hebbian_model.py --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl --split train --validation-split validation --seed 123 --node-count 2048 --max-active-nodes 48 --max-train-rows 26000 --max-validation-rows 804 --max-train-patterns 2000000 --max-validation-patterns 100000 --text-window-bytes 256 --text-window-stride-bytes 64 --epochs 1 --batch-size 512 --learning-rate 0.01 --recall-at-k 32 --device rocm --output results/IF7h_hebbian_trained_model_d5_2m_windows.json --experiment-id IF7h_hebbian_trained_model_d5_2m_windows
```

The result record reports seed 123, the resolved command, hypothesis, git commit `1a32048fb32b1fa32dbf5c2ce4610adb217fafd5`, hardware summary, completion status, and metrics in `results/IF7h_hebbian_trained_model_d5_2m_windows.json`.

## Data Scale

- Train window patterns: 1,977,521.
- Validation window patterns: 96,855.
- Text window: 256 bytes.
- Window stride: 64 bytes.
- D5 rows scanned for training: 26,000 row cap.
- D5 validation rows loaded: 804 row cap.
- Node count: 2,048.
- Hebbian memory updates: 1,977,521.
- Accounted Hebbian storage: 52,837,501 bytes, including label index bytes.

This is about 4.0x larger than the previous 494,403-window IF7d run and about 241x larger than the original 8,190-pattern IF7 mechanism probe.

## Results

| Method | loss | hit@32 | MRR | coverage@32 |
| --- | ---: | ---: | ---: | ---: |
| cue_only | 0.07049910973538846 | 0.848206081255485 | 0.35581504451642326 | 0.12477288063742185 |
| cue_plus_hebbian | 0.07394220776824022 | 0.8378916937690362 | 0.31415555873313583 | 0.11418557581726607 |
| raw_hebbian_memory | n/a | 0.8150121315368334 | 0.19123699619246215 | 0.10161364719498588 |

`hebbian_conditioning_improves_trained_model=false`.

Runtime: 323,996 ms.

## Interpretation

The larger trained run resolves the scale objection for this specific integration path. The current dense cue-plus-Hebbian conditioning design still loses to cue-only after scaling to nearly 2 million train patterns and nearly 100,000 validation patterns.

The IF7 mechanism remains useful as a sparse associative recall signal, but the tested model design is not the right way to wire it into a trained predictor. The next IF7 direction should be a task-aware sparse retrieval/source-linking architecture that uses co-firing memory as candidate generation or context, not a dense full-score feature concatenation.

## Limitations

- This is a trained multilabel hashed-node predictor, not a decoder language model.
- It predicts hashed identifier/import/path nodes rather than generated code tokens.
- It uses a 26,000-row training cap with overlapping windows rather than every D5 train row.
- It is a one-epoch first-pass run.
- It does not test executable code generation, repair, API reuse, or security gates.
