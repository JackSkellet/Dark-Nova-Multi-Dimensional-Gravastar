# D4 Tokenizer Efficiency Probe

Date: 2026-06-21

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compare_tokenizers.py --corpus-jsonl data/hf_mirror/exploratory_d3/corpus.jsonl --target-vocab-size 512 --max-train-texts 64 --max-eval-documents 64 --output results/D4_tokenizer_efficiency_probe.json --experiment-id D4_tokenizer_efficiency_probe --seed 123
```

This is a bounded D4 probe, not the final D5 tokenizer comparison. It trains the simple byte-pair tokenizer on 64 D4 train documents and compares token counts on up to 64 documents per split. The byte baseline is UTF-8 bytes plus EOS.

| Split | Byte Tokens | BPE Tokens | Token Reduction |
| --- | ---: | ---: | ---: |
| train | 774,886 | 355,440 | 2.1801x |
| validation | 1,174,484 | 685,816 | 1.7125x |
| test | 735,566 | 416,682 | 1.7653x |

The trained byte-pair tokenizer reached the requested 512 vocabulary size with 255 merges and checksum `3b6c972b3147c36cafe381ab8bb9d4de9141bc67896ea838e0ff75515999e3dc`.

Limitations:

- Token efficiency only; no model has been trained with this tokenizer.
- Simple in-repo byte-pair trainer, not SentencePiece unigram or a production BPE trainer.
- D4 is JavaScript-only and smaller than the planned D5 corpus.
- Throughput, context coverage under trained models, loss per byte/character, and functional quality remain pending.
