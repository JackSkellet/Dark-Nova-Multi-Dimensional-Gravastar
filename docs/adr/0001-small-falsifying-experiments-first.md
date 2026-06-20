# ADR 0001: Small Falsifying Experiments First

## Status

Accepted.

## Context

The project goal combines routing, compression, selective precision, lookup, and continual learning. A large integrated system would be expensive and could hide failures behind confounders.

## Decision

Start with deterministic CPU-compatible component experiments and compare each idea against simple baselines and random controls. Build an integrated prototype only if component results justify it.

## Consequences

The first results are not claims about large language model performance. They are scaffolding and falsification checks that make later experiments reproducible.

