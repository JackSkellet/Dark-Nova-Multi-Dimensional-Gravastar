# Experiment Plan

## E1: Contextual Routing

Compare static token routing, flat contextual routing, hierarchical contextual routing, and random routing on synthetic polysemous examples. Record route accuracy, entropy, expert usage, active components, estimated traffic, and routing latency.

Follow-up E1b evaluates mixed-topic and adversarial-cue inputs. Compare a single-label contextual router, an explicit multi-label keyword router, and a learned-latent centroid multi-label router. Record exact set accuracy, route recall, mixed-route recall, adversarial exact-set accuracy, false positive routes, active components, estimated traffic, and routing latency.

## E2: Compositional Storage

Compare FP32, FP16, int8, int4, low-rank SVD, low-rank sparse residuals, shared bases, tensorized blocks, Kronecker rank-1, Tensor Train 4D, and a speculative rank-1 outer-product representation on a representative matrix. Record encoded bytes, metadata bytes, reconstruction error, encode/decode time, and effective bits per parameter.

Add product-quantized rows as a codebook baseline. Count codebook values, packed assignments, shape/split metadata, and reconstruction error; do not count it as compression when codebook overhead exceeds the saved value bytes.

## E3: Carrier Discovery

Use a linear task with known important features. Rank candidate carriers by activation magnitude, gradient proxy, causal ablation, and random control. Test precision policies with protected carriers and sparse residuals.

Follow-up E3b tests the failed H3 assumption more directly: use groupwise int4 quantization and select FP32 protected channels by measured training-output error, then evaluate on held-out validation data against random protected sets at the same byte budget.

Follow-up E3c moves from a linear task to a deterministic tiny transformer-style language model. It quantizes the output-head rows, selects protected rows on calibration prompts, and evaluates logit MSE and KL divergence on disjoint held-out prompts.

Follow-up E3d trains the tiny transformer's output head with ridge regression on a deterministic next-token task, then repeats output-head row quantization and protected-row selection on held-out prompts. This remains a toy model, but it avoids relying only on random logits.

Follow-up E3e uses the same trained tiny-transformer output-head setup to explicitly compare selected BF16-like protected carrier rows against selected FP32 protected carrier rows and random protected-row controls. This tests whether precision above BF16 provides material value under matched byte accounting.

Follow-up E3f moves selective precision from the output head to the internal MLP output matrix `w_mlp_out`. Keep the trained output head fixed, quantize the internal matrix, select protected internal rows by train-prompt output error, and evaluate held-out logits and targets against random protected-row controls.

Follow-up E3g trains the internal MLP output matrix itself by ridge regression on the next-token toy task before quantization. Quantize that trained internal matrix, compare groupwise int4 with selected BF16-like and FP32 protected rows, and keep matched random protected-row controls at the same byte budgets.

Follow-up E3h moves from toy tensors to a real open-model checkpoint matrix. Use an explicit command to download a pinned public checkpoint, load only a local PyTorch state dict in reusable code, select a real 2D floating matrix, and compare uniform int4, row-wise groupwise int4, selected BF16-like protected rows, selected FP32 protected rows, and matched random protected-row controls. This first real-matrix run measures reconstruction error only, not task loss.

Follow-up E3i adds a minimal local GPT-2 forward pass for the same pinned tiny checkpoint. Quantize `transformer.wte.weight`, select protected rows by calibration KL on deterministic token-id sequences, and evaluate held-out next-token cross-entropy, KL to the FP32 model, accuracy, and storage. This tests task behavior, but it is still not a natural-language benchmark.

Follow-up E3j replaces E3i's synthetic token-id sequences with byte-level GPT-2 BPE tokenized natural-language repository/code prose using pinned public tokenizer files from the same model repository. Quantize the same `transformer.wte.weight` matrix and compare full FP32, uniform int4, groupwise int4, selected BF16-like/FP32 protected rows, and matched random protected rows on held-out natural-text next-token NLL/KL. This remains a tiny-model CPU smoke test, not useful-scale language-model evidence.

Follow-up E3k keeps the E3j natural-text task but moves from the skinny embedding matrix to an internal MLP tensor, using an all-row candidate strategy rather than token-id-indexed rows. Compare the same precision policies and storage accounting on `transformer.h.0.mlp.c_fc.weight`. This tests whether a wider internal row layout changes the storage-quality conclusion on the pinned tiny model.

## E4: Component Lookup

Create component banks of increasing size and compare dense top-k, exact vector search, centroid tree, hash routing, and cached lookup. Record recall, index memory, cache hit rate, p50, and p95 latency.

Follow-up E4b simulates a fuller routed execution path for larger component banks. It records authorization filtering, lookup, cache, transfer, reconstruction, and dispatch stage timings, plus recall, index memory, component bytes, and cache hit rate.

Follow-up E4c runs a small Torch-backed batched exact-routing path. It applies authorization masking, selects components, moves selected components to the execution device, reconstructs route-specific tensors, and dispatches a batched matrix multiply. It requests ROCm through PyTorch HIP, labels the logical backend separately from PyTorch's internal `cuda` device type, and records backend/fallback status.

Follow-up E4d isolates ROCm transfer scaling from the routed lookup path. It measures host-to-device transfer, a simple device dispatch, and device-to-host transfer across multiple payload sizes. It records ROCm backend/runtime status and explicitly marks that it does not measure occupancy, kernel fusion, power, or useful model-layer work.

## E5: Chronological Continual Learning

Replay synthetic API facts over time. Compare frozen memory, continuously updated retrieval, adapter without replay, adapter with replay, and rollback. Record new/prior accuracy, forgetting, storage growth, lineage, and rollback result.

Add an explicit public-repository changed-file chronology attempt using a local clone:

```bash
git clone https://github.com/antirez/kilo data/raw/public/kilo
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/kilo --max-commits 12 --top-k 5 --include-symbol-qa --include-stale-docs
```

This evaluates future changed-file retrieval from commit subjects using only files and commit mappings available before the future commit. It is not patch generation or neural adapter learning.

Add a symbol-definition QA variant with:

```bash
uv run python scripts/run_public_repo_chronology.py --repo-path data/raw/public/kilo --max-commits 12 --top-k 5 --include-symbol-qa --include-stale-docs
```

The QA task asks which file defines visible symbols using only state available at each exposed commit. New future symbols are tracked separately and are not treated as answerable before exposure.

The stale-document task compares adjacent commits, detects removed symbols, and flags documentation files that still mention those removed symbols at the current commit. The larger current public sample is `pallets/markupsafe` over the last 30 first-parent commits.

Follow-up E5c adds an actual trainable toy adapter path rather than adapter-like memory only. It trains a low-rank parameter delta over chronological API facts with and without replay, compares it with continuously updated retrieval at every step, records new-task and prior-task accuracy, counts weight updates, and includes explicit storage accounting for the retrieval table versus adapter weights.

Follow-up E5d generates a synthetic Git chronology with an API rename, one intentionally stale documentation commit, and a later documentation fix. It runs the same stale-doc detector used for public repositories and records positive-control accuracy, so public zero-positive scans are not the only stale-doc evidence.

Follow-up E5e generates a synthetic repository bug with executable tests and evaluates candidate unified diffs by applying them and running the test suite. This creates a functional patch-generation grading path without using textual similarity to a reference patch.

Follow-up E5f evaluates a public historical fix from a local public clone. The exposed state is the parent commit, a no-op candidate is graded as the frozen baseline, deterministic generated candidates are produced from exposed source when they match known narrow patterns, and the future commit diff is applied as a historical control. Current generated candidates cover a proxy-unsafe `s.__class__ is str` exact check and a Python `clamp(value, low, high)` function that only applies an upper bound. Candidates are accepted only if an executable regression command fails before the patch and passes after applying the candidate. The current real public task uses `pallets/markupsafe` commit `54bb00b`, which fixes proxy-object compatibility in `escape`; the clamp candidate is covered by synthetic public-history regression tests but has not yet matched a real public commit in the local clones.

Follow-up E5g aggregates multiple public historical patch replays into one suite. The initial suite uses the MarkupSafe proxy-object fix from E5f plus `antirez/kilo` commit `8e9a9bb`, which frees `saved_hl` in `FIND_RESTORE_HL`. The MarkupSafe task uses a functional Python regression. The Kilo task uses an executable source-level regression that checks the `memcpy`, `free(saved_hl)`, and `saved_hl = NULL` ordering inside the macro; it is not a runtime leak harness. Report task-level generated-candidate success, historical-patch success, and baseline failure counts.

## E6: Integrated Design

Deferred until E1-E5 justify combination.

## E6a: Structured External Repository Memory

Before integrating a new weight architecture, test an alternative family that does not require continual weight mutation. Compare a frozen parametric proxy, updated text retrieval, gated text retrieval, and gated structured external memory on synthetic repository-convention questions. Include trusted and untrusted records, public and security-scoped records, current and stale conventions, storage accounting, latency, answer accuracy, poisoned-answer count, unauthorized-denial accuracy, and an explicit flag that this is not model-level code-generation evidence.

## E6b: Public Repository Symbol-Memory QA

Move the structured-memory alternative from synthetic convention lookup to local public repository source. Extract visible symbol-definition file mappings from approved public Git clones, ask which file defines each symbol, and compare a frozen parametric proxy, text file retrieval, and structured symbol memory. Record pinned public commit IDs, repo count, symbol/question count, answer accuracy, storage bytes, latency, and limitations. This remains symbol-file QA only, not code generation or natural-language reasoning.

## E6c: Public Repository Signature-Memory QA

Move one step beyond symbol-file lookup by asking for defining signature lines from local public repository source. Extract visible Python/C-like function signatures from approved public Git clones, compare a frozen parametric proxy, text signature lookup, and structured signature memory, and record pinned commits, repo count, signature/question count, answer accuracy, storage bytes, latency, and limitations. This remains signature-line retrieval only, not semantic code understanding or generation.

## E6d: Public Repository Call-Stub Generation

Use the extracted Python signatures as a constrained code-generation proxy. Generate canonical call stubs from public Python functions, compare a frozen parametric proxy, a name-only call-stub baseline, and structured signature memory, and record pinned commits, repo count, function/question count, answer accuracy, storage bytes, latency, and limitations. This tests whether structured memory can drive simple syntax-aware generation from repository source, but it does not synthesize executable arguments or semantic program behavior.

## E6e: Public Repository Function-Skeleton Generation

Move one step beyond call expressions while staying within a measurable reduced task. Generate minimal Python function skeletons from public Python signatures, compare a frozen parametric proxy, a name-only skeleton baseline, and structured signature memory, and record pinned commits, repo count, function/question count, answer accuracy, storage bytes, latency, and limitations. This tests whether structured memory can preserve repository signatures in generated code scaffolds, but the body is a deterministic placeholder and does not synthesize executable behavior or semantic implementation logic.

## E6f: Public Repository Docstring-Skeleton Generation

Add a documentation-aware reduced generation task. Extract public Python functions with docstrings, generate minimal function skeletons that preserve the signature and first docstring line, compare a frozen parametric proxy, a signature-only skeleton baseline, and structured docstring memory, and record pinned commits, repo count, function/question count, answer accuracy, storage bytes, latency, and limitations. This tests whether structured memory can carry lightweight code-documentation contracts into generated scaffolds, but it only preserves the first docstring line and does not synthesize executable behavior or semantic implementation logic.

## E6g: Public Repository API-Reference Generation

Move from code scaffolds to documentation generation while staying measurable. Extract public Python functions with signatures, source file paths, and first docstring lines, generate Markdown API reference entries, compare a frozen parametric proxy, a signature-only API reference baseline, and structured docstring memory, and record pinned commits, repo count, function/question count, answer accuracy, storage bytes, latency, and limitations. This tests whether structured memory can generate lightweight repository API documentation entries, but it uses a deterministic template and does not judge documentation quality or synthesize semantic explanations.

## E6h: Public Repository API-Doc Coverage QA

Move from generated API entries to source-documentation consistency. Extract public Python functions and Sphinx-style documentation directives from local public repositories, ask which documentation file covers each API symbol, compare a frozen parametric proxy, a source-symbol-only memory baseline, and structured documentation-directive memory, and record pinned commits, repo count, symbol/question count, answer accuracy, storage bytes, latency, and limitations. This tests exact API documentation coverage lookup, but it does not judge documentation quality or detect semantic drift in prose.

## E6i: Synthetic API-Doc Drift Detection

Move from exact coverage lookup to a first source/documentation inconsistency positive control. Generate a local synthetic Git history where a Python API symbol is renamed in source while a Sphinx `autofunction` directive still references the old symbol, then ask whether the documented API is consistent with current source. Compare a frozen parametric proxy, a doc-directive-only memory baseline, and structured source/documentation consistency memory. Record the synthetic commit, question count, stale-doc issue count, answer accuracy, storage bytes, latency, and limitations. This tests API-doc drift detection as a controlled source/docs consistency task, but it is synthetic and only detects missing source symbols behind directives, not documentation quality or semantic prose drift.

## S1: Security Canary And Access Control

Create synthetic scoped documents, synthetic canary secrets, and stored prompt-injection text. Verify deterministic authorization happens before retrieval/context construction, canary documents are quarantined, prompt-injection documents are quarantined, and every decision is audit logged. This is a smoke test, not a production security proof.

## S2: Adversarial Security Suite

Exercise the same retrieval and update-controller boundaries with synthetic adversarial cases: repo-scope bypass requests, explicit canary-extraction requests, stored prompt-injection variants, poisoning/backdoor text in candidate data, poisoned update candidates, tampered update payloads, and rollback. Success requires zero unauthorized retrievals, zero canary extraction, zero prompt-injection bypasses, zero poisoned or tampered update deployments, and audit events for every decision. This remains a reduced adversarial suite, not a production red-team.

## S2b: Obfuscated Extraction Corpus

Extend S2 with separator-obfuscated synthetic canaries, base64/hex encoded synthetic canaries, paraphrased stored-instruction text, poisoning text, and repeated bypass/extraction queries. Success requires zero canary leakage in allowed contexts, zero unauthorized retrievals, quarantine of every encoded canary document, quarantine of every paraphrased stored-instruction document, and audit events covering every document decision. This tests normalization and basic encoding handling, not semantic robustness against novel attacks.

## S2c: Semantic Extraction Red-Team Smoke Test

Extend S2b with synthetic sensitive aliases that do not use the `DN_CANARY_*` pattern, trust-label quarantine for known sensitive documents, and meaning-equivalent extraction/exfiltration queries such as summarizing, translating, encoding, or sending forbidden markers. Success requires every sensitive alias document to be quarantined by trust label, every semantic exfiltration query to be denied before retrieval, zero sensitive-alias leakage in allowed contexts, zero unauthorized retrievals, and at least two clean queries still returning authorized clean documents. This is a deterministic reduced heuristic and does not prove real semantic robustness.

## E5b: External Update Controller

Simulate the externally controlled Tier 2 update path for a repository adapter. A candidate update must carry training-data lineage, configuration lineage, checksum, signature, external approval, new-task score, prior-task regression score, and security/privacy test results. The controller should accept only candidates that improve over continuously updated retrieval, keep prior-task regression within the pre-registered 2% absolute limit, have zero critical security failures, verify checksum/signature integrity, write audit events, deploy atomically, and roll back successfully.
