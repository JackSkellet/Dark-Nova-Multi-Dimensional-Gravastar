# D5 Fast-BPE Trained Model Comparison

Date: 2026-06-21

This comparison trains the same dense-528 decoder on D5 with either the byte tokenizer or the fast `tokenizers` byte-level BPE artifact `artifacts/D5_fast_bpe_8192_seed123/tokenizer.json`.

All runs use ROCm, FP32, AdamW, learning rate 0.0001, seed 123, validation seed 424242, sequence length 128, batch size 2, 3 layers, 8 heads, hidden size 528, `block_impl=explicit_causal`, and `attention_mask_mode=finite_causal`.

## Runs

| Run | Tokenizer | Protocol | Train token-units | Validation loss/token | Validation estimated nats/byte | Validation exact nats/raw byte | Validation exact bits/raw byte | Token-units/s | Estimated train bytes/s | Peak VRAM | Model-only bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `D5_byte_dense528_seed123_5m_equal_compute` | byte | 5,000,192 byte tokens | 5,000,192 | 1.573074267245829 | 1.5732627631603064 | 1.5845346788925965 | 2.286000323354935 | 22,590.893185151985 | 22,587.88984629075 | 463,684,096 | 41,605,253 |
| `D5_bpe8192_dense528_seed123_5m_equal_compute` | BPE-8192 | 5,000,192 BPE tokens | 5,000,192 | 4.371953308349475 | 1.4991149059415514 | 1.4745071520290007 | 2.127264155987548 | 16,442.099762019545 | 50,711.25136100197 | 710,005,760 | 75,370,501 |
| `D5_bpe8192_dense528_seed123_equal_raw_bytes` | BPE-8192 | about 5,000,970 estimated raw bytes | 1,621,248 | 5.166740634478629 | 1.771642411067963 | 1.7425611560964318 | 2.5139843383460607 | 16,018.114033067333 | 49,403.580979142396 | 711,112,704 | 75,370,501 |

## Tokenization And Context Coverage

Full D5 train tokenization:

- Byte tokenizer: 200,000,495 tokens over 199,973,906 bytes.
- BPE-8192 tokenizer: 64,837,503 tokens over 199,973,906 bytes.
- Train token reduction: 3.0846421553279124x.
- Validation token reduction: 2.9167058286796284x.

At the same 128-token context length, estimated validation byte coverage rises from 127.9846640512837 bytes for byte tokens to 373.29361561998326 bytes for BPE tokens. This is useful context coverage, but it also changes the raw-byte span per attention window, so it is not identical compute per raw byte.

## Interpretation

The BPE-8192 equal-compute run improves validation loss per estimated byte by 0.07414785721875505 nats/byte versus the byte-tokenizer baseline and improves exact evaluated loss by 0.11002752686359574 nats/raw byte. It has about 2.245x estimated train-byte throughput. It uses a larger embedding/output matrix, raising model-only size from about 41.6 MB to 75.4 MB and peak allocated VRAM from about 464 MB to 710 MB.

The equal-raw-byte BPE run is worse than the byte baseline by 0.1983796479076565 estimated nats/byte and 0.15802647720383534 exact nats/raw byte. This means token reduction alone is not a win: the model also needs enough optimizer steps/token updates to use the larger tokenizer effectively.

Exact raw-byte accounting comes from separate checkpoint evaluations that record every evaluated target token's NLL and decoded byte length, then divide total target NLL by evaluated raw target bytes. The byte run evaluated 131,072 target tokens / 130,124 raw target bytes; both BPE runs evaluated 131,072 target tokens / 388,632 raw target bytes on the same validation sampler.

Functional quality was not measured in this comparison. The next tokenizer decision should use executable D5 code tasks or another task-sensitive functional benchmark, not only token loss.

## Source Artifacts

- `artifacts/D5_fast_bpe_8192_seed123/tokenizer.json`
- `results/D5_fast_bpe_8192_seed123_tokenizer.json`
- `results/D5_byte_dense528_seed123_5m_equal_compute.json`
- `results/D5_bpe8192_dense528_seed123_5m_equal_compute.json`
- `results/D5_bpe8192_dense528_seed123_equal_raw_bytes.json`
- `results/D5_byte_dense528_seed123_5m_equal_compute_exact_validation_eval.json`
- `results/D5_bpe8192_dense528_seed123_5m_equal_compute_exact_validation_eval.json`
- `results/D5_bpe8192_dense528_seed123_equal_raw_bytes_exact_validation_eval.json`
- `results/D5_tokenizer_training_comparison.json`
