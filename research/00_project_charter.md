# Project Charter

## Mission

Determine whether any secure, fully local AI-weight architecture improves the Pareto frontier for code-specialized inference and adaptation compared with strong simpler baselines.

## Non-Goals For Initial Pass

- No full pretraining.
- No billion-parameter continual-learning experiment.
- No private or confidential data ingestion.
- No production serving.
- No novelty claims without source review and evidence.

## Operating Rules

Treat every proposed idea as unproven. State null hypotheses first, include baselines and random controls, and preserve negative results. Compression claims must count total encoded bytes, including metadata and temporary buffers. Speed claims must be measured end-to-end, not inferred from operation counts.

## Initial Scope

The repository starts with reduced synthetic experiments for H1-H5, a primary-source review seed, and a secure on-prem architecture design. Later scale-up must pass a design-selection gate.

