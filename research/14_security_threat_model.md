# Security Threat Model

## Assets

Confidential source code, internal documentation, credentials, private keys, customer information, generated patches, model weights, adapters, retrieval indexes, training buffers, evaluation data, update metadata, and audit logs.

## Threats

Data poisoning, model poisoning, backdoors, prompt injection stored in documents, malicious code comments, dependency compromise, secret memorization, training-data extraction, membership inference, cross-repository leakage, unauthorized retrieval, tampered indexes, tampered adapters, malicious checkpoints, rollback attacks, synthetic-data feedback loops, and compromised update evaluation.

## Controls

- Deterministic authorization before retrieval and before context construction.
- Repository-level scopes and role/attribute-based access control.
- Trust-label quarantine for known sensitive local documents.
- Secret scanning and canary tests using synthetic canaries only.
- Query-level denial for reduced semantic exfiltration requests.
- Signed model, adapter, and index artifacts.
- Checksummed indexes and immutable audit records.
- Fail-closed network policy for production runtime.
- Reproducible offline installation.
- Rollback and data-deletion rebuild procedures.

Any critical confidentiality or authorization failure overrides average quality gains.

## S1 Synthetic Canary Result

Status: generated from `results/S1_security_canary_access_control.json`.

The S1 experiment creates synthetic documents across two repository scopes, including:

- one authorized clean document,
- one unauthorized cross-repository document,
- one synthetic canary document,
- one stored prompt-injection document.

Measured result:

- Unauthorized retrievals: 0.
- Canary leaks in allowed context: 0.
- Blocked secret documents: 1.
- Prompt-injection documents quarantined: 1.
- Audit events: 4.
- Critical failures: 0.

Interpretation: the prototype demonstrates deterministic authorization before retrieval, canary quarantine before context construction, prompt-injection quarantine, and audit logging. It is not production proof: there is no OS sandbox, cryptographic signing, real identity integration, adversarial prompt suite, training-data extraction test, or side-channel analysis.

## E5b Update-Gate Security Result

Status: generated from `results/E5b_external_update_controller.json`.

The E5b controller rejects candidate adapter artifacts with checksum mismatch, signature mismatch, self approval, missing lineage, regression over the 2% limit, no improvement over retrieval, unauthorized retrievals, canary leakage, or poisoning findings. The current run rejects one canary-leaking candidate and one tampered artifact before deployment, accepts one clean externally approved candidate, and successfully rolls back to the base version. This covers update tampering detection and rollback success as a prototype gate, but it does not replace real key management, artifact signing, sandboxed training, or production audit-log storage.

## S2 Adversarial Security Suite

Status: generated from `results/S2_adversarial_security_suite.json`.

S2 extends S1 with synthetic adversarial prompts and data:

- a request to ignore repository scopes and retrieve a different repository,
- a request to print a synthetic canary,
- stored prompt-injection variants using ignore-all and system-override phrasing,
- stored poisoning/backdoor text that asks to disable authorization,
- a poisoned adapter-update candidate,
- a tampered adapter-update candidate.

Measured result:

- Access-control bypass successes: 0.
- Canary extraction successes: 0.
- Prompt-injection bypass successes: 0.
- Poisoned update acceptances: 0.
- Tampered update acceptances: 0.
- Blocked prompt-injection documents: 3.
- Blocked poisoned documents: 1.
- Rejected poisoning candidates: 1.
- Critical failures: 0.

Interpretation: this is a stronger adversarial smoke test than S1 because it explicitly tests bypass requests, canary-extraction attempts, poisoning text, update poisoning, and payload tampering. It is still pattern-based and synthetic; it does not prove resistance to novel prompt-injection encodings, real training-data extraction, side channels, malicious dependencies, or compromised identity systems.

## S2b Obfuscated Extraction Corpus

Status: generated from `results/S2b_adversarial_extraction_corpus.json`.

S2b extends S2 with:

- separator-obfuscated synthetic canaries,
- base64 and hex encoded synthetic canaries,
- paraphrased stored-instruction text using spaced or hyphenated words,
- poisoning text with update-approval language,
- repeated bypass and extraction queries.

Measured result:

- Queries: 12.
- Documents: 12.
- Obfuscated/encoded canary documents: 4.
- Encoded canary documents quarantined: 4.
- Paraphrased prompt-injection documents: 4.
- Paraphrased prompt-injection documents quarantined: 4.
- Poisoned documents quarantined: 1.
- Unauthorized retrievals: 0.
- Canary leaks in allowed context: 0.
- Prompt-injection bypass successes: 0.
- Poisoning bypass successes: 0.
- Audit events: 144.
- Critical failures: 0.

Interpretation: this reduces one S2 weakness by testing normalized separator handling and simple encoded canary forms rather than only direct phrases. It remains deterministic pattern coverage, not a production red-team: it does not test semantic aliases without explicit trust labels, multilingual attacks, image/PDF channels, real model extraction, sandbox escapes, side channels, or compromised identity systems.

## S2c Semantic Extraction Red-Team Smoke Test

Status: generated from `results/S2c_semantic_extraction_red_team.json`.

S2c extends S2b with:

- sensitive aliases that do not contain the `DN_CANARY_*` string pattern,
- trust-label quarantine for known sensitive documents,
- meaning-equivalent extraction requests that ask to summarize, translate, encode, list, or send forbidden markers,
- clean authorized queries that should still retrieve non-sensitive local context.

Measured result:

- Queries: 8.
- Documents: 6.
- Semantic secret documents: 3.
- Trust-label quarantines: 3.
- Semantic exfiltration queries: 6.
- Semantic exfiltration queries blocked: 6.
- Clean queries allowed: 2.
- Unauthorized retrievals: 0.
- Sensitive alias leaks in allowed context: 0.
- Audit events: 18.
- Critical failures: 0.

Interpretation: this moves beyond literal canary matching by enforcing metadata/trust labels and blocking a small reduced set of meaning-equivalent exfiltration requests. It is still not production semantic security: the query classifier is deterministic, English-only, and hand-authored; aliases are protected because the documents carry trusted quarantine labels; there is no real model extraction, multilingual red-team, file-format attack, sandbox test, side-channel analysis, or identity-system integration.
