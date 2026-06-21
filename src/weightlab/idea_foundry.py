from __future__ import annotations

import json
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IDEA_FOUNDRY_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "IF1",
        "name": "Repository graph conditioned decoder",
        "family": "code_structure",
        "uses_adapters": False,
        "uses_moe_or_topic_routing": False,
        "addresses": ["code_structure", "functional_evaluation"],
        "novelty_label": "adjacent",
        "mechanism": (
            "Add typed repository-edge bias and optional edge prediction losses to a "
            "dense causal decoder. Edges are imports, exports, tests, docs, and "
            "same-symbol file links resolved inside a repository-aware split."
        ),
        "equations": [
            "A_ij = (Q_i K_j^T / sqrt(d)) + causal_ij + b_type(edge(i,j))",
            "L = L_lm + lambda_edge * CE(edge_type_hat, edge_type)",
        ],
        "closest_primary_source_prior_art": [
            {
                "title": "GraphCodeBERT: Pre-training Code Representations with Data Flow",
                "url": "https://arxiv.org/abs/2009.08366",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "GraphCodeBERT uses data-flow structure in encoder pre-training; "
                    "IF1 tests repository-level import/test/doc edges inside a small "
                    "causal decoder and keeps split-aware evaluation fixed."
                ),
            }
        ],
        "expected_scaling": (
            "Bias lookup is O(edges in window) and should be negligible versus dense "
            "attention at 128-token context; preprocessing scales linearly with files."
        ),
        "rocm_plan": (
            "Start with additive attention bias tensors in the existing explicit causal "
            "block; keep FP32 AdamW and finite masks before any fused-kernel work."
        ),
        "likely_failure": (
            "D5 rows may not expose enough resolvable repository graph structure, and "
            "file-level edges may not land inside short 128-token windows often enough."
        ),
        "cheapest_falsifying_test": (
            "Scan D5 for import/test/doc edges, split leakage, and local edge resolution. "
            "Reject the pilot if resolvable edges are too sparse or cross split boundaries."
        ),
        "mechanism_occurrence_evidence": (
            "D5 rows contain repo, path, split, role, and source text fields that can "
            "support import/test/doc edge extraction without fetching external data."
        ),
    },
    {
        "id": "IF2",
        "name": "Signed fast-weight evolution scratchpad",
        "family": "continual_evolution",
        "uses_adapters": False,
        "uses_moe_or_topic_routing": False,
        "addresses": ["continual_evolution"],
        "novelty_label": "adjacent",
        "mechanism": (
            "Keep base weights fixed during a session and update a small associative "
            "fast-weight matrix from verified local examples. Promotion to persistent "
            "state requires regression and security gates."
        ),
        "equations": [
            "F_t = decay * F_{t-1} + eta * phi(k_t) phi(v_t)^T",
            "h'_t = h_t + gate(h_t) * F_t phi(q_t)",
        ],
        "closest_primary_source_prior_art": [
            {
                "title": "Using Fast Weights to Attend to the Recent Past",
                "url": "https://arxiv.org/abs/1610.06258",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "Fast weights store recent sequence memories; IF2 makes the memory "
                    "repository-scoped, signed, security-gated, and explicitly compared "
                    "against retrieval and replay."
                ),
            }
        ],
        "expected_scaling": (
            "Associative matrix cost is O(d*r) per step for rank r; storage is bounded "
            "by explicit eviction and signed promotion policies."
        ),
        "rocm_plan": (
            "Prototype as small FP32/BF16 matrix multiplies beside the dense block, then "
            "benchmark whether HIP GEMM overhead beats retrieval latency."
        ),
        "likely_failure": (
            "Fast weights may memorize surface tokens without improving held-out repair "
            "or API reuse, and poisoning gates may reject most useful updates."
        ),
        "cheapest_falsifying_test": (
            "Chronological API-rename fixture: compare retrieval-only, fast-weight-only, "
            "and retrieval-plus-fast-weight answers after one verified update."
        ),
        "mechanism_occurrence_evidence": (
            "Earlier E5 chronology fixtures already contain controlled update events and "
            "stale-document failures suitable for a one-update fast-weight proxy."
        ),
    },
    {
        "id": "IF3",
        "name": "Audited block-codebook weight generator",
        "family": "compression",
        "uses_adapters": False,
        "uses_moe_or_topic_routing": False,
        "addresses": ["weight_storage_compression"],
        "novelty_label": "potentially_novel",
        "mechanism": (
            "Represent each trained weight block by a compact codebook id, scale, and "
            "small sparse residual. A deterministic generator reconstructs blocks from "
            "layer id, block coordinates, and codebook entries; all metadata is counted."
        ),
        "equations": [
            "W_l,b ~= s_l,b * C[z_l,b] + U_l,b V_l,b^T + Sparse(R_l,b)",
            "bytes = bytes(C) + bytes(z,s,U,V,R,indexes,reconstruction_cache)",
        ],
        "closest_primary_source_prior_art": [
            {
                "title": (
                    "GPTQ: Accurate Post-Training Quantization for "
                    "Generative Pre-trained Transformers"
                ),
                "url": "https://arxiv.org/abs/2210.17323",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "GPTQ quantizes existing weights; IF3 tests reconstructible block "
                    "generation plus residual accounting and random-control codebooks."
                ),
            },
            {
                "title": "Tensorizing Neural Networks",
                "url": "https://arxiv.org/abs/1509.06569",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "Tensor train layers factorize tensors; IF3 uses audited learned "
                    "block codebooks plus sparse/low-rank residuals for trained decoders."
                ),
            },
        ],
        "expected_scaling": (
            "Storage can fall below per-channel int8 only if codebook ids and residuals "
            "stay small; reconstruction adds O(block_size) unless cached."
        ),
        "rocm_plan": (
            "First reconstruct into dense FP32/BF16 tensors for loss checks, then test a "
            "packed lookup/dequant kernel only if metadata-inclusive storage wins."
        ),
        "likely_failure": (
            "Metadata and reconstruction caches may erase storage gains, or random "
            "codebook controls may match learned codebooks."
        ),
        "cheapest_falsifying_test": (
            "Apply block-codebook reconstruction to T11c/T12 dense checkpoints and "
            "compare loss, bytes, and random-codebook controls."
        ),
        "mechanism_occurrence_evidence": (
            "Existing quantization records already show uniform int8 is strong and int4 "
            "is weak, creating a clear compression baseline to beat."
        ),
    },
    {
        "id": "IF4",
        "name": "Executable trace contrastive objective",
        "family": "code_document_objective",
        "uses_adapters": False,
        "uses_moe_or_topic_routing": False,
        "addresses": ["code_structure", "documentation", "functional_evaluation"],
        "novelty_label": "adjacent",
        "mechanism": (
            "Train with ordinary language modeling plus contrastive alignment between "
            "source spans, tests, docs, signatures, and observed executable traces."
        ),
        "equations": [
            "L = L_lm + lambda * InfoNCE(e_source, e_test_or_doc, negatives)",
            "score(file, task) = dot(pool(source), pool(trace_or_doc))",
        ],
        "closest_primary_source_prior_art": [
            {
                "title": (
                    "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder "
                    "Models for Code Understanding and Generation"
                ),
                "url": "https://arxiv.org/abs/2109.00859",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "CodeT5 uses identifier-aware code/text objectives; IF4 requires "
                    "executable test traces and source-doc drift negatives as paired signals."
                ),
            }
        ],
        "expected_scaling": (
            "Auxiliary pooling is linear in sampled paired spans; executable trace "
            "collection is the expensive offline step."
        ),
        "rocm_plan": (
            "Share the dense-528 trunk, add small projection heads, and keep the "
            "auxiliary loss in FP32 until the baseline loss path is unchanged."
        ),
        "likely_failure": (
            "D5 may not include runnable environments, and weak filename/docstring pairs "
            "may add noise rather than useful task signal."
        ),
        "cheapest_falsifying_test": (
            "Build a tiny public-repo fixture with source, tests, docs, and API drift; "
            "verify file selection and signature accuracy beat source-only scoring."
        ),
        "mechanism_occurrence_evidence": (
            "D5 role labels include source, tests, README, documentation, docstrings, "
            "and changelogs, although roles overlap and timestamps are mostly absent."
        ),
    },
    {
        "id": "IF5",
        "name": "Syntax-state recurrent mixer",
        "family": "state_space_hybrid",
        "uses_adapters": False,
        "uses_moe_or_topic_routing": False,
        "addresses": ["long_context", "code_structure"],
        "novelty_label": "adjacent",
        "mechanism": (
            "Replace one transformer block with a selective recurrent mixer whose state "
            "is explicitly modulated by indentation, bracket depth, comment state, and "
            "string-literal state."
        ),
        "equations": [
            "s_t = A(x_t, syntax_t) s_{t-1} + B(x_t, syntax_t) x_t",
            "h_t = h_t + C(x_t, syntax_t) s_t",
        ],
        "closest_primary_source_prior_art": [
            {
                "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
                "url": "https://arxiv.org/abs/2312.00752",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "Mamba makes SSM parameters input-selective; IF5 adds explicit "
                    "code-syntax state features and tests source/span tasks rather than "
                    "generic long-context claims."
                ),
            }
        ],
        "expected_scaling": (
            "Linear in sequence length for the mixer block; total model remains mostly "
            "transformer cost until more blocks are replaced."
        ),
        "rocm_plan": (
            "Start with eager PyTorch recurrent scan for correctness, then port the scan "
            "to a HIP-friendly chunked kernel only if the pilot beats dense loss/function."
        ),
        "likely_failure": (
            "At 128-token context the recurrent state may add overhead without enough "
            "long-range structure benefit."
        ),
        "cheapest_falsifying_test": (
            "Measure bracket/comment/string-state prediction and causal span recovery on "
            "D5 before any full 50M-token training."
        ),
        "mechanism_occurrence_evidence": (
            "D5 source files include syntax delimiters, comments, minified files, tests, "
            "and docs where explicit lexical state can be computed without labels."
        ),
    },
    {
        "id": "IF6",
        "name": "Delta-consolidated specialist lattice",
        "family": "continual_specialization",
        "uses_adapters": True,
        "uses_moe_or_topic_routing": True,
        "addresses": ["continual_evolution", "repository_specialization"],
        "novelty_label": "adjacent",
        "mechanism": (
            "Keep a stable dense core and train small repository/language specialists, "
            "then periodically consolidate specialists into shared deltas only when "
            "held-out validation, security, and regression gates pass."
        ),
        "equations": [
            "h' = h + sum_{k in topK(g(repo, task))} alpha_k Delta_k(h)",
            "Delta_shared = argmin_D sum_k ||Delta_k - D c_k||_2 + beta * regressions",
        ],
        "closest_primary_source_prior_art": [
            {
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "url": "https://arxiv.org/abs/2106.09685",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "LoRA trains low-rank adapters; IF6 adds repository-aware specialist "
                    "lattices, consolidation gates, rollback, and retrieval/memory controls."
                ),
            },
            {
                "title": (
                    "Switch Transformers: Scaling to Trillion Parameter Models with "
                    "Simple and Efficient Sparsity"
                ),
                "url": "https://arxiv.org/abs/2101.03961",
                "access_date": "2026-06-21",
                "exact_difference": (
                    "Switch routes tokens to experts; IF6 routes signed local deltas by "
                    "repository/task and requires chronological continual-learning tests."
                ),
            },
        ],
        "expected_scaling": (
            "Training scales with active specialists; inference overhead depends on top-k "
            "deltas and consolidation frequency."
        ),
        "rocm_plan": (
            "Reuse the residual-adapter branch only as a control, then test specialist "
            "delta matmuls with explicit top-k dispatch and dense fallback."
        ),
        "likely_failure": (
            "Specialists may overfit repository identities, and routing/consolidation "
            "metadata may exceed any quality or storage gain."
        ),
        "cheapest_falsifying_test": (
            "Chronological multi-repo API reuse fixture with retrieval-only, dense, "
            "adapter, and specialist-lattice controls."
        ),
        "mechanism_occurrence_evidence": (
            "D5 has repository IDs, language labels, and role labels needed to define "
            "specialist scopes while preserving repository-aware splits."
        ),
    },
]


_IMPORT_PATTERNS = (
    re.compile(r"\bimport\s+(?:[^'\"]+\s+from\s+)?['\"](?P<target>[^'\"]+)['\"]"),
    re.compile(r"\brequire\(\s*['\"](?P<target>[^'\"]+)['\"]\s*\)"),
    re.compile(r"\bexport\s+[^'\"]*from\s+['\"](?P<target>[^'\"]+)['\"]"),
)


def summarize_candidate_constraints(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "without_adapters": sum(not row["uses_adapters"] for row in candidates),
        "without_moe_or_topic_routing": sum(
            not row["uses_moe_or_topic_routing"] for row in candidates
        ),
        "continual_evolution_candidates": sum(
            "continual_evolution" in row["addresses"] for row in candidates
        ),
        "compression_candidates": sum(
            "weight_storage_compression" in row["addresses"] for row in candidates
        ),
        "code_structure_candidates": sum(
            "code_structure" in row["addresses"] for row in candidates
        ),
        "potentially_novel_candidates": sum(
            row["novelty_label"] == "potentially_novel" for row in candidates
        ),
        "candidate_ids": [row["id"] for row in candidates],
    }


def run_repository_graph_signal_probe(
    corpus_jsonl: Path,
    *,
    max_documents: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    repo_splits: dict[str, set[str]] = defaultdict(set)
    repo_paths: dict[str, set[str]] = defaultdict(set)
    role_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    with corpus_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_documents is not None and index >= max_documents:
                break
            row = json.loads(line)
            rows.append(row)
            repo = str(row.get("repo", ""))
            path = _normalize_path(str(row.get("path", "")))
            repo_splits[repo].add(str(row.get("split", "unknown")))
            repo_paths[repo].add(path)
            language_counts[str(row.get("language", "unknown"))] += 1
            for role in _ordered_unique_roles(row.get("content_roles", ["source"])):
                role_counts[role] += 1

    edges: list[dict[str, Any]] = []
    resolved_edges: list[dict[str, Any]] = []
    for row in rows:
        repo = str(row.get("repo", ""))
        source_path = _normalize_path(str(row.get("path", "")))
        for target in _extract_import_targets(str(row.get("text", ""))):
            edge = {
                "repo": repo,
                "source_path": source_path,
                "target": target,
                "split": str(row.get("split", "unknown")),
            }
            edges.append(edge)
            resolved = _resolve_relative_target(source_path, target, repo_paths[repo])
            if resolved is not None:
                resolved_edge = dict(edge)
                resolved_edge["resolved_path"] = resolved
                resolved_edges.append(resolved_edge)

    repos_with_edges = {edge["repo"] for edge in edges}
    repositories_in_multiple_splits = {
        repo: sorted(splits) for repo, splits in repo_splits.items() if len(splits) > 1
    }
    return {
        "benchmark_label": "idea_foundry_repository_graph_signal_probe",
        "candidate_id": "IF1",
        "corpus_jsonl": str(corpus_jsonl),
        "document_count": len(rows),
        "language_counts": dict(sorted(language_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "import_edge_count": len(edges),
        "resolved_local_edge_count": len(resolved_edges),
        "repositories_with_edges": len(repos_with_edges),
        "repository_aware_splits_preserved": not repositories_in_multiple_splits,
        "repositories_in_multiple_splits": len(repositories_in_multiple_splits),
        "edge_examples": resolved_edges[:20],
        "mechanism_signal_present": bool(edges and resolved_edges),
        "limitations": [
            "regex_import_extraction_only",
            "no_ast_or_package_resolution",
            "edge_presence_probe_not_model_training",
        ],
    }


def _extract_import_targets(text: str) -> list[str]:
    targets: list[str] = []
    for pattern in _IMPORT_PATTERNS:
        targets.extend(match.group("target") for match in pattern.finditer(text))
    return targets


def _resolve_relative_target(
    source_path: str,
    target: str,
    repo_paths: set[str],
) -> str | None:
    if not target.startswith("."):
        return None
    base = posixpath.dirname(source_path)
    raw = _normalize_path(posixpath.normpath(posixpath.join(base, target)))
    candidates = [
        raw,
        f"{raw}.js",
        f"{raw}.jsx",
        f"{raw}.ts",
        f"{raw}.tsx",
        posixpath.join(raw, "index.js"),
        posixpath.join(raw, "index.ts"),
    ]
    for candidate in candidates:
        if candidate in repo_paths:
            return candidate
    return None


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return "" if normalized == "." else normalized.lstrip("/")


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
