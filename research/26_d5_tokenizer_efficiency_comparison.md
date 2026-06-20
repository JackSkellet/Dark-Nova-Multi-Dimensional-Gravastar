# D5 Tokenizer Efficiency Comparison

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compare_tokenizers.py --corpus-jsonl data/hf_mirror/exploratory_d5_balanced/corpus.jsonl --target-vocab-size 512 --max-train-texts 64 --max-eval-documents 256 --context-lengths 128 512 1024 2048 --output results/D5_tokenizer_efficiency_comparison.json --experiment-id D5_tokenizer_efficiency_comparison --seed 123
```

The trained byte-pair tokenizer reached 512 vocabulary entries with 255 merges and checksum `fea22cbb81ed72b78025708a49075efbeb2e99872edd0b1040e6bb41bea4759e`.

| Split | Byte Tokens | BPE Tokens | Token Reduction |
| --- | ---: | ---: | ---: |
| train | 3,196,506 | 1,855,447 | 1.7228x |
| validation | 3,334,650 | 2,105,917 | 1.5835x |
| test | 2,732,930 | 1,649,367 | 1.6570x |

## Context Coverage

Mean fraction of document bytes covered by a fixed token window:

| Split | Tokenizer | 128 tok | 512 tok | 1024 tok | 2048 tok |
| --- | --- | ---: | ---: | ---: | ---: |
| train | byte | 0.0379 | 0.1517 | 0.3035 | 0.5125 |
| train | BPE | 0.0658 | 0.2626 | 0.4618 | 0.6656 |
| validation | byte | 0.0420 | 0.1679 | 0.3358 | 0.5628 |
| validation | BPE | 0.0707 | 0.2826 | 0.5093 | 0.7127 |
| test | byte | 0.0620 | 0.2079 | 0.3653 | 0.5609 |
| test | BPE | 0.0991 | 0.3215 | 0.5135 | 0.7078 |

## Throughput

The byte tokenizer is trivial UTF-8 byte encoding and runs hundreds of MB/s on this sample. The current BPE implementation is a simple pure-Python merge loop and is much slower:

| Split | BPE bytes/s | BPE tokens/s |
| --- | ---: | ---: |
| train | 118,984.9 | 69,071.6 |
| validation | 111,009.7 | 70,110.9 |
| test | 114,602.8 | 69,171.1 |

An attempted larger 1024-vocabulary / 256-train-document run was terminated after more than six minutes because the current pure-Python trainer was too slow for interactive iteration. That is an implementation bottleneck, not evidence against the tokenizer objective.

## Limitations

- This is a bounded D5 tokenizer-efficiency comparison, not model training.
- Loss per byte/character and functional quality require matched byte-tokenizer and BPE model runs.
- The simple BPE trainer is not SentencePiece unigram and is not optimized.
- Context coverage estimates use token-count fraction as a byte-coverage proxy for BPE.
