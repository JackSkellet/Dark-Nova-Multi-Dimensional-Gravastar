from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from weightlab.compression import compare_compression_methods
from weightlab.config import default_config_path, read_accelerator_backend
from weightlab.continual import (
    run_chronological_memory_experiment,
    run_trainable_adapter_vs_retrieval_experiment,
)
from weightlab.external_memory import (
    run_public_repository_api_doc_coverage_qa_experiment,
    run_public_repository_api_reference_generation_experiment,
    run_public_repository_call_stub_generation_experiment,
    run_public_repository_docstring_skeleton_generation_experiment,
    run_public_repository_function_skeleton_generation_experiment,
    run_public_repository_memory_qa_experiment,
    run_public_repository_signature_qa_experiment,
    run_structured_repository_memory_experiment,
    run_synthetic_api_doc_drift_detection_experiment,
)
from weightlab.importance import (
    evaluate_carrier_importance,
    evaluate_output_error_selected_precision,
    evaluate_tiny_transformer_precision,
    evaluate_trained_internal_layer_precision,
    evaluate_trained_tiny_transformer_bf16_vs_fp32_carriers,
    evaluate_trained_tiny_transformer_internal_matrix_precision,
    evaluate_trained_tiny_transformer_precision,
)
from weightlab.lookup import (
    benchmark_lookup_methods,
    benchmark_rocm_transfer_scaling,
    benchmark_routed_execution,
    benchmark_torch_batched_routed_execution,
)
from weightlab.metrics import ExperimentRecord, write_json
from weightlab.repo_chronology import (
    run_synthetic_patch_generation_positive_control,
    run_synthetic_stale_doc_positive_control,
)
from weightlab.routing import evaluate_contextual_routing, evaluate_routing_robustness
from weightlab.security import (
    run_adversarial_extraction_corpus_experiment,
    run_adversarial_security_experiment,
    run_security_canary_experiment,
    run_semantic_extraction_red_team_experiment,
)
from weightlab.update_controller import simulate_update_controller_experiment


def _existing_result_records(
    output_dir: Path, current_experiment_ids: set[str]
) -> list[dict[str, object]]:
    preserved: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in {"manifest.json", "research_assessment.json"}:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        experiment_id = str(record.get("experiment_id", ""))
        if experiment_id and experiment_id not in current_experiment_ids:
            preserved.append(record)
    return preserved


def run_all(
    seed: int, output_dir: Path, accelerator_backend: str = "rocm"
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    matrix = (rng.normal(size=(64, 64)) @ rng.normal(size=(64, 64))).astype(np.float32)

    experiments = [
        ExperimentRecord(
            experiment_id="E1_contextual_routing",
            hypothesis="H1",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_contextual_routing(seed=seed, n_samples=1000),
            notes=(
                "Synthetic polysemy task with static, flat contextual, hierarchical, "
                "and random routers."
            ),
        ),
        ExperimentRecord(
            experiment_id="E1b_routing_robustness",
            hypothesis="H1",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_routing_robustness(seed=seed, n_samples=600),
            notes=(
                "Synthetic mixed-topic and adversarial-cue routing with single-label, "
                "explicit multi-label, and latent-centroid multi-label routers."
            ),
        ),
        ExperimentRecord(
            experiment_id="E2_compositional_storage",
            hypothesis="H2",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics={"methods": compare_compression_methods(matrix, rank=8)},
            notes="Synthetic representative matrix; includes metadata in byte counts.",
        ),
        ExperimentRecord(
            experiment_id="E3_carrier_importance",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_carrier_importance(
                seed=seed, n_samples=1024, n_features=48, protected_count=6
            ),
            notes="Linear causal carrier task with random-protection and sparse-residual controls.",
        ),
        ExperimentRecord(
            experiment_id="E3b_output_error_selected_precision",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_output_error_selected_precision(
                seed=seed,
                n_train=1024,
                n_validation=1024,
                n_features=48,
                protected_count=6,
                group_size=8,
            ),
            notes=(
                "Groupwise int4 and validation-output-error-selected FP32 carrier "
                "protection with random controls."
            ),
        ),
        ExperimentRecord(
            experiment_id="E3c_tiny_transformer_precision",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_tiny_transformer_precision(
                seed=seed,
                n_calibration_prompts=64,
                n_heldout_prompts=64,
                protected_count=6,
            ),
            notes=(
                "Tiny transformer-style LM output-head precision allocation evaluated on "
                "held-out prompts."
            ),
        ),
        ExperimentRecord(
            experiment_id="E3d_trained_tiny_transformer_precision",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_trained_tiny_transformer_precision(
                seed=seed,
                n_train_prompts=96,
                n_heldout_prompts=64,
                protected_count=6,
            ),
            notes=(
                "Ridge-trained tiny transformer-style LM output-head precision allocation "
                "evaluated on held-out prompts."
            ),
        ),
        ExperimentRecord(
            experiment_id="E3e_bf16_vs_fp32_carriers",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_trained_tiny_transformer_bf16_vs_fp32_carriers(
                seed=seed,
                n_train_prompts=96,
                n_heldout_prompts=64,
                protected_count=6,
            ),
            notes=(
                "Ridge-trained tiny transformer-style LM comparing selected BF16-like "
                "carrier rows with selected FP32 carrier rows and random controls."
            ),
        ),
        ExperimentRecord(
            experiment_id="E3f_internal_matrix_precision",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_trained_tiny_transformer_internal_matrix_precision(
                seed=seed,
                n_train_prompts=96,
                n_heldout_prompts=64,
                protected_count=8,
            ),
            notes=(
                "Ridge-trained tiny transformer-style LM with selective precision on "
                "the internal MLP output matrix rather than the output head."
            ),
        ),
        ExperimentRecord(
            experiment_id="E3g_trained_internal_layer_precision",
            hypothesis="H3",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=evaluate_trained_internal_layer_precision(
                seed=seed,
                n_train_prompts=96,
                n_heldout_prompts=64,
                protected_count=8,
            ),
            notes=(
                "Tiny transformer-style LM with the internal MLP output matrix trained "
                "by ridge regression before selective precision is evaluated."
            ),
        ),
        ExperimentRecord(
            experiment_id="E4_component_lookup",
            hypothesis="H4",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=benchmark_lookup_methods(
                seed=seed, bank_sizes=[16, 128, 1024], n_queries=256, dim=32
            ),
            notes="CPU isolated lookup benchmark; complete routed execution remains future work.",
        ),
        ExperimentRecord(
            experiment_id="E4b_routed_execution",
            hypothesis="H4",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=benchmark_routed_execution(
                seed=seed,
                bank_sizes=[128, 2048, 8192],
                n_queries=384,
                dim=32,
                component_dim=64,
                repeat_fraction=0.5,
            ),
            notes=(
                "Larger-bank routed execution simulation with authorization, cache, "
                "transfer, reconstruction, and dispatch timing."
            ),
        ),
        ExperimentRecord(
            experiment_id="E4c_torch_batched_routed_execution",
            hypothesis="H4",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=benchmark_torch_batched_routed_execution(
                seed=seed,
                bank_sizes=[512, 2048],
                n_queries=128,
                dim=32,
                component_dim=48,
                batch_size=32,
                device=accelerator_backend,
            ),
            notes=(
                "Torch-backed batched exact routed execution with authorization masking, "
                "component movement, reconstruction, and batched dispatch. Requests ROCm "
                "through PyTorch HIP and records backend and fallback status."
            ),
        ),
        ExperimentRecord(
            experiment_id="E4d_rocm_transfer_scaling",
            hypothesis="H4",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=benchmark_rocm_transfer_scaling(
                payload_bytes=[1 << 20, 8 << 20, 32 << 20],
                iterations=8,
                warmup_iterations=2,
                device=accelerator_backend,
            ),
            notes=(
                "ROCm-requested Torch transfer scaling benchmark with separate "
                "host-to-device, device dispatch, and device-to-host timing. This "
                "does not measure occupancy, power, fusion, or useful model layers."
            ),
        ),
        ExperimentRecord(
            experiment_id="E5_chronological_continual_learning",
            hypothesis="H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_chronological_memory_experiment(seed=seed),
            notes=(
                "Synthetic chronological API-fact replay with retrieval, adapter replay, "
                "and rollback."
            ),
        ),
        ExperimentRecord(
            experiment_id="E5c_trainable_adapter_vs_retrieval",
            hypothesis="H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_trainable_adapter_vs_retrieval_experiment(seed=seed),
            notes=(
                "Toy trainable low-rank adapter delta compared with continuously "
                "updated retrieval under chronological API-fact replay."
            ),
        ),
        ExperimentRecord(
            experiment_id="E5d_synthetic_stale_doc_positive_control",
            hypothesis="H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_synthetic_stale_doc_positive_control(top_k=1),
            notes=(
                "Generated Git-history positive control for stale-document detection "
                "after an API rename and later documentation fix."
            ),
        ),
        ExperimentRecord(
            experiment_id="E5e_synthetic_patch_generation_functional",
            hypothesis="H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_synthetic_patch_generation_positive_control(),
            notes=(
                "Generated patch-generation positive control graded by applying unified "
                "diffs and running unit tests, not text similarity."
            ),
        ),
        ExperimentRecord(
            experiment_id="S1_security_canary_access_control",
            hypothesis="security",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_security_canary_experiment(),
            notes=(
                "Synthetic canary, authorization-before-retrieval, prompt-injection "
                "quarantine, and audit-log simulation."
            ),
        ),
        ExperimentRecord(
            experiment_id="E5b_external_update_controller",
            hypothesis="H5/security",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=simulate_update_controller_experiment(),
            notes=(
                "Synthetic external controller for versioned adapter deployment, "
                "quality/regression gates, checksum/signature tamper detection, "
                "security gates, audit logging, and rollback."
            ),
        ),
        ExperimentRecord(
            experiment_id="S2_adversarial_security_suite",
            hypothesis="security/H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_adversarial_security_experiment(),
            notes=(
                "Synthetic adversarial suite for access-control bypass prompts, "
                "canary extraction, stored prompt injection, poisoning/backdoor text, "
                "poisoned update rejection, tamper detection, and rollback."
            ),
        ),
        ExperimentRecord(
            experiment_id="S2b_adversarial_extraction_corpus",
            hypothesis="security/H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_adversarial_extraction_corpus_experiment(),
            notes=(
                "Synthetic adversarial corpus for obfuscated canaries, paraphrased stored "
                "instructions, poisoning text, and repeated extraction/bypass queries."
            ),
        ),
        ExperimentRecord(
            experiment_id="S2c_semantic_extraction_red_team",
            hypothesis="security/H5",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_semantic_extraction_red_team_experiment(),
            notes=(
                "Synthetic semantic red-team corpus for sensitive aliases, trust-label "
                "quarantine, and meaning-equivalent exfiltration queries."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6a_structured_repository_memory",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_structured_repository_memory_experiment(seed=seed),
            notes=(
                "Reduced alternative-architecture experiment comparing frozen "
                "parametric, text-retrieval, gated text-retrieval, and structured "
                "external-memory approaches for repository convention QA."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6b_public_repository_memory_qa",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_memory_qa_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_symbols_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for symbol "
                "definition file QA, compared with text file retrieval and a frozen "
                "parametric proxy. Uses local public clones when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6c_public_repository_signature_qa",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_signature_qa_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_signatures_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for defining "
                "signature-line QA, compared with text signature lookup and a frozen "
                "parametric proxy. Uses local public clones when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6d_public_repository_call_stub_generation",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_call_stub_generation_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_functions_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for canonical "
                "Python call-stub generation from signatures, compared with a name-only "
                "stub baseline and a frozen parametric proxy. Uses local public clones "
                "when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6e_public_repository_function_skeleton_generation",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_function_skeleton_generation_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_functions_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for minimal "
                "Python function-skeleton generation from signatures, compared with "
                "a name-only skeleton baseline and a frozen parametric proxy. Uses "
                "local public clones when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6f_public_repository_docstring_skeleton_generation",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_docstring_skeleton_generation_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_functions_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for minimal "
                "Python function-skeleton generation that preserves a signature and "
                "first docstring line, compared with a signature-only skeleton "
                "baseline and a frozen parametric proxy. Uses local public clones "
                "when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6g_public_repository_api_reference_generation",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_api_reference_generation_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_functions_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for Markdown "
                "API reference generation from public Python signatures, source file "
                "paths, and first docstring lines, compared with a signature-only "
                "reference baseline and a frozen parametric proxy. Uses local public "
                "clones when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6h_public_repository_api_doc_coverage_qa",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_public_repository_api_doc_coverage_qa_experiment(
                repo_paths=[
                    Path("data/raw/public/kilo"),
                    Path("data/raw/public/markupsafe"),
                ],
                max_symbols_per_repo=64,
            ),
            notes=(
                "Public-repository structured external-memory benchmark for API "
                "documentation coverage QA from Sphinx directives, compared with a "
                "source-symbol-only memory baseline and a frozen parametric proxy. "
                "Uses local public clones when present."
            ),
        ),
        ExperimentRecord(
            experiment_id="E6i_synthetic_api_doc_drift_detection",
            hypothesis="A1_external_memory",
            seed=seed,
            command=f"uv run python scripts/run_experiments.py --seed {seed}",
            metrics=run_synthetic_api_doc_drift_detection_experiment(),
            notes=(
                "Synthetic Git-history positive control for API documentation drift: "
                "a source symbol is renamed while a Sphinx autofunction directive "
                "still references the old API. Compares doc-directive-only memory "
                "with structured source/documentation consistency memory."
            ),
        ),
    ]
    jsonable = [record.to_jsonable() for record in experiments]
    current_experiment_ids = {str(record["experiment_id"]) for record in jsonable}
    preserved = _existing_result_records(output_dir, current_experiment_ids)
    for record in jsonable:
        write_json(output_dir / f"{record['experiment_id']}.json", record)
    manifest_records = sorted(
        preserved + jsonable,
        key=lambda record: (
            float(record.get("recorded_at_unix", 0.0)),
            str(record.get("experiment_id", "")),
        ),
    )
    write_json(output_dir / "manifest.json", manifest_records)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["experiment_id", "hypothesis", "seed", "status", "notes"]
        )
        writer.writeheader()
        for record in manifest_records:
            writer.writerow({k: record[k] for k in writer.fieldnames})
    return manifest_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--accelerator-backend", choices=["rocm", "cuda", "cpu"])
    args = parser.parse_args()
    accelerator_backend = (
        args.accelerator_backend or read_accelerator_backend(args.config) or "rocm"
    )
    run_all(
        seed=args.seed,
        output_dir=args.output_dir,
        accelerator_backend=accelerator_backend,
    )


if __name__ == "__main__":
    main()
