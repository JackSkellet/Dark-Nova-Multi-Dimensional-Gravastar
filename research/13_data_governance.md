# Data Governance

## Trust Levels

- High: human-reviewed merged code, passing tests, signed releases, approved documentation, authoritative specifications.
- Medium: open pull requests, issues, review discussions, build logs, static-analysis reports.
- Low: raw model outputs, unreviewed patches, untrusted documents, code comments containing instructions, arbitrary prompts.

## Ingestion Gate

Every artifact must pass authorization, scope assignment, sensitivity classification, secret scanning, license/ownership checks, duplicate detection, poisoning/backdoor screening, instruction/data separation, quality scoring, quarantine, validation, and immutable lineage recording.

No raw interaction should go directly into weight training.

## Provenance

Track human-authored, model-authored, mixed, human-approved, test-validated, rejected, and synthetic replay data separately. Never treat model-generated text as independent ground truth.

## D5 Materialization Gate

The exploratory D5 materializer keeps the primary training split repository-aware by stable repository hash. It also records an independent temporal split label when row-level timestamp columns such as `commit_date`, `created_at`, `updated_at`, or `last_modified` are present. Temporal labels are therefore auditable metadata until timestamp coverage is measured on the materialized corpus.

Duplicate filtering has two stages: exact SHA256 over normalized text, then deterministic 64-bit SimHash over normalized token shingles with band lookup and Hamming-distance confirmation. This is a practical near-duplicate screen for the exploratory stream, not a full MinHash/LSH deduplication proof.

Each accepted row also receives multi-label content-role metadata for source, tests, README, documentation, docstrings, and changelogs. D5 acceptance is not complete until the materialized record reports role counts and token counts, so gaps in tests/docs/changelogs are visible rather than hidden inside aggregate token totals.
