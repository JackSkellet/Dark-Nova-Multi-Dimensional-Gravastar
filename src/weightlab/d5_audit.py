from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def audit_d5_corpus(
    corpus_jsonl: Path,
    *,
    near_duplicate_hamming_threshold: int = 3,
    near_duplicate_min_bytes: int = 200,
    example_limit: int = 20,
) -> dict[str, Any]:
    token_mass = {
        "by_language": defaultdict(_mass_counter),
        "by_source": defaultdict(_mass_counter),
        "by_repository": defaultdict(_mass_counter),
        "by_role": defaultdict(_mass_counter),
    }
    role_sets: dict[str, dict[str, int]] = defaultdict(_mass_counter)
    role_pair_counts: Counter[str] = Counter()
    exact_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    simhash_rows: list[tuple[int, dict[str, Any]]] = []
    repo_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    repo_split_mismatches = 0
    temporal_split_counts: Counter[str] = Counter()
    timestamps_by_source: Counter[str] = Counter()
    rows = 0
    rows_with_timestamp = 0
    rows_without_timestamp = 0
    total_tokens = 0
    role_token_sum = 0
    multi_role_rows = 0

    with corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            tokens = int(row.get("tokens", 0))
            total_tokens += tokens
            split = str(row.get("split", "unknown"))
            repo_split = str(row.get("repo_split", split))
            repo = str(row.get("repo", ""))
            source = _source_key(row)
            roles = _ordered_unique_roles(row.get("content_roles", ["source"]))
            role_set = "+".join(roles)
            timestamp = str(row.get("source_timestamp", ""))

            _add_mass(token_mass["by_language"][str(row.get("language", "unknown"))], tokens)
            _add_mass(token_mass["by_source"][source], tokens)
            _add_mass(token_mass["by_repository"][repo], tokens)
            _add_mass(role_sets[role_set], tokens)
            for role in roles:
                _add_mass(token_mass["by_role"][role], tokens)
                role_token_sum += tokens
            if len(roles) > 1:
                multi_role_rows += 1
                for left_index, left in enumerate(roles):
                    for right in roles[left_index + 1 :]:
                        role_pair_counts[f"{left}+{right}"] += 1

            split_counts[split] += 1
            split_tokens[split] += tokens
            repo_splits[repo].add(split)
            if repo_split != split:
                repo_split_mismatches += 1
            temporal_split_counts[str(row.get("temporal_split", "unknown"))] += 1
            if timestamp:
                rows_with_timestamp += 1
                timestamps_by_source[source] += 1
            else:
                rows_without_timestamp += 1

            exact_groups[str(row.get("text_sha256", ""))].append(
                {
                    "dataset": str(row.get("dataset", "")),
                    "config": str(row.get("config", "")),
                    "repo": repo,
                    "path": str(row.get("path", "")),
                    "split": split,
                    "source": source,
                }
            )
            simhash_value = str(row.get("near_duplicate_simhash64", ""))
            if simhash_value and int(row.get("bytes", 0)) >= near_duplicate_min_bytes:
                simhash_rows.append(
                    (
                        int(simhash_value, 16),
                        {
                            "source": source,
                            "repo": repo,
                            "path": str(row.get("path", "")),
                            "split": split,
                            "bytes": int(row.get("bytes", 0)),
                        },
                    )
                )

    exact_duplicate_groups = [
        group for digest, group in exact_groups.items() if digest and len(group) > 1
    ]
    near_duplicate = _near_duplicate_summary(
        simhash_rows,
        threshold=near_duplicate_hamming_threshold,
        min_bytes=near_duplicate_min_bytes,
        example_limit=example_limit,
    )
    repositories_in_multiple_splits = {
        repo: sorted(splits) for repo, splits in repo_splits.items() if len(splits) > 1
    }
    temporal_claim_allowed = (
        rows > 0
        and rows_without_timestamp == 0
        and bool(temporal_split_counts)
        and "unknown" not in temporal_split_counts
    )
    return {
        "benchmark_label": "d5_corpus_audit",
        "corpus_jsonl": str(corpus_jsonl),
        "rows": rows,
        "tokens": total_tokens,
        "token_mass": {
            "by_language": _sort_mass(token_mass["by_language"]),
            "by_source": _sort_mass(token_mass["by_source"]),
            "by_repository": _sort_mass(token_mass["by_repository"]),
            "by_role": _sort_mass(token_mass["by_role"]),
        },
        "role_overlap": {
            "multi_role_rows": multi_role_rows,
            "role_sets": _sort_mass(role_sets),
            "role_pair_rows": dict(sorted(role_pair_counts.items())),
            "role_token_sum": role_token_sum,
            "role_token_sum_exceeds_total_tokens": role_token_sum > total_tokens,
        },
        "split_integrity": {
            "split_counts": dict(sorted(split_counts.items())),
            "split_tokens": dict(sorted(split_tokens.items())),
            "repo_split_mismatches": repo_split_mismatches,
            "repositories_in_multiple_splits": len(repositories_in_multiple_splits),
            "repositories_in_multiple_splits_examples": dict(
                list(sorted(repositories_in_multiple_splits.items()))[:example_limit]
            ),
            "repository_aware_splits_preserved": (
                repo_split_mismatches == 0 and not repositories_in_multiple_splits
            ),
        },
        "timestamp_coverage": {
            "rows_with_timestamp": rows_with_timestamp,
            "rows_without_timestamp": rows_without_timestamp,
            "coverage_ratio": rows_with_timestamp / rows if rows else 0.0,
            "by_source_rows_with_timestamp": dict(sorted(timestamps_by_source.items())),
            "temporal_split_counts": dict(sorted(temporal_split_counts.items())),
            "temporal_claim_allowed": temporal_claim_allowed,
        },
        "duplicates": {
            "exact_text": _exact_duplicate_summary(exact_duplicate_groups, example_limit),
            "near_duplicate_simhash": near_duplicate,
        },
        "limitations": [
            "near_duplicate_audit_uses_simhash_threshold_not_full_suffix_or_minhash_search",
            "timestamp_audit_reports_existing_rows_only_and_does_not_fetch_external_metadata",
            "repository_mass_can_be_skewed_by_large_files_and_is_not_quality_weighting",
        ],
    }


def _mass_counter() -> dict[str, int]:
    return {"rows": 0, "tokens": 0}


def _add_mass(counter: dict[str, int], tokens: int) -> None:
    counter["rows"] += 1
    counter["tokens"] += tokens


def _source_key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset', '')}::{row.get('config', '')}"


def _ordered_unique_roles(raw_roles: Any) -> list[str]:
    roles: list[str] = []
    seen: set[str] = set()
    for raw_role in raw_roles:
        role = str(raw_role)
        if role in seen:
            continue
        roles.append(role)
        seen.add(role)
    return roles or ["source"]


def _sort_mass(mapping: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        key: value
        for key, value in sorted(
            mapping.items(),
            key=lambda item: (-item[1]["tokens"], -item[1]["rows"], item[0]),
        )
    }


def _exact_duplicate_summary(
    duplicate_groups: list[list[dict[str, str]]],
    example_limit: int,
) -> dict[str, Any]:
    cross_source = [
        group for group in duplicate_groups if len({row["source"] for row in group}) > 1
    ]
    cross_split = [
        group for group in duplicate_groups if len({row["split"] for row in group}) > 1
    ]
    return {
        "duplicate_groups": len(duplicate_groups),
        "duplicate_rows": sum(len(group) - 1 for group in duplicate_groups),
        "cross_source_duplicate_groups": len(cross_source),
        "cross_split_duplicate_groups": len(cross_split),
        "examples": duplicate_groups[:example_limit],
    }


def _near_duplicate_summary(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    threshold: int,
    min_bytes: int,
    example_limit: int,
) -> dict[str, Any]:
    by_band: dict[int, list[int]] = defaultdict(list)
    candidate_pairs = 0
    cross_source_pairs = 0
    cross_split_pairs = 0
    examples: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for index, (simhash, metadata) in enumerate(rows):
        candidates: set[int] = set()
        for band in _simhash_bands(simhash):
            candidates.update(by_band.get(band, []))
        for candidate_index in candidates:
            pair_key = (candidate_index, index)
            if pair_key in seen_pairs:
                continue
            previous_hash, previous_metadata = rows[candidate_index]
            distance = (simhash ^ previous_hash).bit_count()
            if distance > threshold:
                continue
            seen_pairs.add(pair_key)
            candidate_pairs += 1
            if metadata["source"] != previous_metadata["source"]:
                cross_source_pairs += 1
            if metadata["split"] != previous_metadata["split"]:
                cross_split_pairs += 1
            if len(examples) < example_limit:
                examples.append(
                    {
                        "hamming_distance": distance,
                        "left": previous_metadata,
                        "right": metadata,
                    }
                )
        for band in _simhash_bands(simhash):
            by_band[band].append(index)
    return {
        "algorithm": "simhash64_band_candidate_audit",
        "hamming_threshold": threshold,
        "min_bytes": min_bytes,
        "candidate_pairs": candidate_pairs,
        "cross_source_candidate_pairs": cross_source_pairs,
        "cross_split_candidate_pairs": cross_split_pairs,
        "examples": examples,
    }


def _simhash_bands(simhash: int) -> tuple[int, int, int, int]:
    mask = (1 << 16) - 1
    return tuple(((band << 16) | ((simhash >> (band * 16)) & mask)) for band in range(4))
