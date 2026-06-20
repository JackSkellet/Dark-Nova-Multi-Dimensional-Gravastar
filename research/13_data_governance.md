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

