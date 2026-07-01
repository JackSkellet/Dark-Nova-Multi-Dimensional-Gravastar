from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weightlab.if7_sparse_hebbian import (
    HebbianTrainingConfig,
    RepositoryLinkingRankerConfig,
    SparseHebbianConfig,
    SparseHebbianMemory,
    SparseHebbianRerankerConfig,
    run_if7_hebbian_repository_linking,
    run_if7_hebbian_sparse_reranker,
    run_if7_hebbian_trained_model,
    run_if7_sparse_hebbian_probe,
    run_if7_trained_repository_linking_ranker,
)


def _write_rows(path: Path) -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "repo": "repo-alpha",
                "path": f"src/config_{index}.js",
                "split": "train",
                "language": "JavaScript",
                "content_roles": ["source"],
                "row_sha256": f"row-alpha-{index}",
                "text_sha256": f"text-alpha-{index}",
                "bytes": 180,
                "text": (
                    "import parser from './parser';\n"
                    f"export function loadConfig{index}(rawOptions) {{\n"
                    "  return parser.normalizeConfig(rawOptions);\n"
                    "}\n"
                ),
            }
        )
        rows.append(
            {
                "repo": "repo-beta",
                "path": f"tests/render_{index}.test.js",
                "split": "train",
                "language": "JavaScript",
                "content_roles": ["test"],
                "row_sha256": f"row-beta-{index}",
                "text_sha256": f"text-beta-{index}",
                "bytes": 190,
                "text": (
                    "import { renderGraph } from '../src/render';\n"
                    f"test('render graph {index}', () => {{\n"
                    "  expect(renderGraph([{ id: 'node' }])).toBeTruthy();\n"
                    "});\n"
                ),
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_sparse_hebbian_memory_strengthens_nodes_that_fire_together():
    config = SparseHebbianConfig(node_count=512, max_active_nodes=32)
    memory = SparseHebbianMemory(config)

    repeated_pattern = [3, 7, 11, 19]
    separate_pattern = [3, 23, 29, 31]
    for _ in range(8):
        memory.observe(repeated_pattern)
    memory.observe(separate_pattern)

    assert memory.connection_strength(3, 7) > memory.connection_strength(3, 23)
    ranked = memory.complete([3], top_k=5)
    assert 7 in ranked
    assert memory.accounted_storage_bytes() > 0


def test_if7_probe_uses_sparse_real_corpus_rows_and_compares_controls(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = run_if7_sparse_hebbian_probe(
        corpus_jsonl=corpus,
        config=SparseHebbianConfig(
            node_count=1024,
            max_active_nodes=40,
            max_train_rows=20,
            max_eval_rows=10,
            recall_at_k=16,
            seed=123,
        ),
    )

    assert result["benchmark_label"] == "if7_sparse_hebbian_assembly_probe"
    assert result["candidate_id"] == "IF7"
    assert result["corpus"]["rows_loaded"] == 20
    assert result["corpus"]["source_split"] == "train"
    assert result["sparsity"]["mean_active_nodes"] < result["config"]["node_count"]
    assert result["methods"]["hebbian_sparse_assembly"]["storage_bytes"] > 0
    assert result["methods"]["random_sparse_control"]["storage_bytes"] > 0
    assert result["final"]["hebbian_hit_at_k"] >= result["final"]["frequency_hit_at_k"]
    assert result["final"]["hebbian_hit_at_k"] >= result["final"]["random_hit_at_k"]
    assert result["hebbian_adds_associative_signal"] is True
    assert result["sample_predictions"]


def test_if7_cli_writes_machine_readable_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "if7.json"
    _write_rows(corpus)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_if7_sparse_hebbian_probe.py",
            "--corpus-jsonl",
            str(corpus),
            "--split",
            "train",
            "--node-count",
            "1024",
            "--max-active-nodes",
            "40",
            "--max-train-rows",
            "20",
            "--max-eval-rows",
            "10",
            "--recall-at-k",
            "16",
            "--output",
            str(output),
            "--experiment-id",
            "IF7_tmp",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "IF7_tmp"
    assert record["status"] == "completed"
    assert record["metrics"]["candidate_id"] == "IF7"
    assert record["metrics"]["benchmark_label"] == "if7_sparse_hebbian_assembly_probe"
    assert "--max-train-rows 20" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_if7_trains_model_with_hebbian_features_on_real_rows(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = run_if7_hebbian_trained_model(
        corpus_jsonl=corpus,
        config=HebbianTrainingConfig(
            split="train",
            validation_split="train",
            node_count=512,
            max_active_nodes=40,
            max_train_rows=18,
            max_validation_rows=6,
            epochs=8,
            batch_size=6,
            learning_rate=5e-2,
            recall_at_k=16,
            device="cpu",
            seed=123,
        ),
    )

    assert result["benchmark_label"] == "if7_hebbian_conditioned_trained_model"
    assert result["candidate_id"] == "IF7"
    assert result["training"]["supervised_train_rows"] == 18
    assert result["training"]["validation_rows"] == 6
    assert result["models"]["cue_only"]["parameter_count"] > 0
    assert result["models"]["cue_plus_hebbian"]["parameter_count"] > result["models"][
        "cue_only"
    ]["parameter_count"]
    assert result["models"]["cue_only"]["loss_history"][0] > result["models"]["cue_only"][
        "loss_history"
    ][-1]
    assert result["models"]["cue_plus_hebbian"]["loss_history"][0] > result["models"][
        "cue_plus_hebbian"
    ]["loss_history"][-1]
    assert result["validation"]["cue_plus_hebbian"]["hit_at_k"] >= result["validation"][
        "cue_only"
    ]["hit_at_k"]
    assert "model_training_not_just_memory_probe" not in result["limitations"]


def test_if7_training_can_expand_rows_into_multiple_text_window_patterns(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {
            "repo": "repo-window",
            "path": "src/large.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "row-window",
            "text_sha256": "text-window",
            "bytes": 1600,
            "text": "\n".join(
                f"export function symbol{i}(input) {{ return normalizeConfig(input) + {i}; }}"
                for i in range(40)
            ),
        }
    ]
    corpus.write_text(json.dumps(rows[0], sort_keys=True) + "\n", encoding="utf-8")

    result = run_if7_hebbian_trained_model(
        corpus_jsonl=corpus,
        config=HebbianTrainingConfig(
            split="train",
            validation_split="train",
            node_count=256,
            max_train_rows=1,
            max_validation_rows=1,
            max_train_patterns=8,
            max_validation_patterns=4,
            text_window_bytes=256,
            text_window_stride_bytes=128,
            epochs=2,
            batch_size=4,
            device="cpu",
            seed=123,
        ),
    )

    assert result["corpus"]["train_rows_loaded"] == 1
    assert result["training"]["supervised_train_patterns"] > result["corpus"][
        "train_rows_loaded"
    ]
    assert result["training"]["supervised_train_patterns"] <= 8
    assert result["training"]["validation_patterns"] <= 4


def test_if7_trained_model_cli_writes_machine_readable_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "if7b.json"
    _write_rows(corpus)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_if7_hebbian_model.py",
            "--corpus-jsonl",
            str(corpus),
            "--split",
            "train",
            "--validation-split",
            "train",
            "--node-count",
            "512",
            "--max-active-nodes",
            "40",
            "--max-train-rows",
            "18",
            "--max-validation-rows",
            "6",
            "--epochs",
            "4",
            "--batch-size",
            "6",
            "--learning-rate",
            "0.05",
            "--recall-at-k",
            "16",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--experiment-id",
            "IF7b_tmp",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "IF7b_tmp"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "if7_hebbian_conditioned_trained_model"
    assert record["metrics"]["training"]["supervised_train_rows"] == 18
    assert "--epochs 4" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_if7_sparse_reranker_trains_on_hebbian_topk_candidates(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = run_if7_hebbian_sparse_reranker(
        corpus_jsonl=corpus,
        config=SparseHebbianRerankerConfig(
            split="train",
            validation_split="train",
            node_count=512,
            max_train_rows=20,
            max_validation_rows=8,
            candidate_count=64,
            max_train_candidates=4096,
            max_validation_patterns=8,
            epochs=6,
            batch_size=128,
            learning_rate=0.05,
            recall_at_k=16,
            device="cpu",
            seed=123,
        ),
    )

    assert result["benchmark_label"] == "if7_sparse_hebbian_candidate_reranker"
    assert result["candidate_id"] == "IF7"
    assert result["training"]["train_patterns"] >= 1
    assert result["training"]["train_candidate_examples"] > result["training"][
        "train_patterns"
    ]
    assert result["models"]["candidate_reranker"]["parameter_count"] < 64
    assert result["models"]["candidate_reranker"]["loss_history"][0] > result["models"][
        "candidate_reranker"
    ]["loss_history"][-1]
    assert result["validation"]["candidate_reranker"]["hit_at_k"] >= result["validation"][
        "raw_hebbian_memory"
    ]["hit_at_k"]
    assert result["validation"]["candidate_recall_ceiling"]["hit_at_k"] >= result[
        "validation"
    ]["candidate_reranker"]["hit_at_k"]


def test_if7_sparse_reranker_cli_writes_machine_readable_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "if7e.json"
    _write_rows(corpus)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_if7_sparse_reranker.py",
            "--corpus-jsonl",
            str(corpus),
            "--split",
            "train",
            "--validation-split",
            "train",
            "--node-count",
            "512",
            "--max-train-rows",
            "20",
            "--max-validation-rows",
            "8",
            "--candidate-count",
            "64",
            "--max-train-candidates",
            "4096",
            "--max-validation-patterns",
            "8",
            "--epochs",
            "4",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.05",
            "--recall-at-k",
            "16",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--experiment-id",
            "IF7e_tmp",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "IF7e_tmp"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "if7_sparse_hebbian_candidate_reranker"
    assert "--candidate-count 64" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_if7_repository_linking_ranks_same_repo_files_against_distractors(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {
            "repo": "repo-alpha",
            "path": "src/config.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "train-alpha-source",
            "bytes": 160,
            "text": "export function loadConfig(raw) { return normalizeConfig(raw); }\n",
        },
        {
            "repo": "repo-alpha",
            "path": "tests/config.test.js",
            "split": "train",
            "language": "JavaScript",
            "content_roles": ["test"],
            "row_sha256": "train-alpha-test",
            "bytes": 150,
            "text": "test('loadConfig normalizes config', () => loadConfig({}));\n",
        },
        {
            "repo": "repo-beta",
            "path": "src/render.js",
            "split": "validation",
            "language": "JavaScript",
            "content_roles": ["source"],
            "row_sha256": "val-beta-source",
            "bytes": 150,
            "text": "export function renderGraph(nodes) { return drawGraph(nodes); }\n",
        },
        {
            "repo": "repo-beta",
            "path": "tests/render.test.js",
            "split": "validation",
            "language": "JavaScript",
            "content_roles": ["test"],
            "row_sha256": "val-beta-test",
            "bytes": 150,
            "text": "test('renderGraph draws graph nodes', () => renderGraph([]));\n",
        },
        {
            "repo": "repo-gamma",
            "path": "docs/config.md",
            "split": "validation",
            "language": "Markdown",
            "content_roles": ["documentation"],
            "row_sha256": "val-gamma-doc",
            "bytes": 120,
            "text": "Configuration documentation for unrelated settings.\n",
        },
    ]
    corpus.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = run_if7_hebbian_repository_linking(
        corpus_jsonl=corpus,
        train_split="train",
        eval_split="validation",
        seed=123,
        node_count=512,
        max_train_rows=10,
        max_eval_repositories=8,
        negatives_per_query=2,
        top_k=2,
    )

    assert result["benchmark_label"] == "if7_hebbian_repository_linking"
    assert result["candidate_id"] == "IF7"
    assert result["corpus"]["eval_repositories"] >= 1
    assert result["tasks"]["task_count"] >= 2
    assert result["methods"]["raw_hebbian_context"]["hit_at_k"] >= 0.0
    assert result["methods"]["combined_lexical_hebbian"]["hit_at_k"] >= result[
        "methods"
    ]["lexical_text_overlap"]["hit_at_k"]
    assert result["sample_tasks"]


def test_if7_repository_linking_cli_writes_machine_readable_record(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)
    output = tmp_path / "if7g.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_if7_repository_linking.py",
            "--corpus-jsonl",
            str(corpus),
            "--train-split",
            "train",
            "--eval-split",
            "train",
            "--node-count",
            "512",
            "--max-train-rows",
            "20",
            "--max-eval-repositories",
            "8",
            "--negatives-per-query",
            "4",
            "--top-k",
            "2",
            "--output",
            str(output),
            "--experiment-id",
            "IF7g_tmp",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "IF7g_tmp"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "if7_hebbian_repository_linking"
    assert "--top-k 2" in record["command"]
    assert (tmp_path / "manifest.json").exists()


def test_if7_trained_repository_linking_ranker_learns_task_aware_scores(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)

    result = run_if7_trained_repository_linking_ranker(
        corpus_jsonl=corpus,
        config=RepositoryLinkingRankerConfig(
            train_split="train",
            eval_split="train",
            seed=123,
            node_count=512,
            max_memory_rows=24,
            max_train_repositories=4,
            max_eval_repositories=4,
            negatives_per_query=4,
            top_k=2,
            epochs=8,
            batch_size=32,
            learning_rate=0.05,
            device="cpu",
        ),
    )

    assert result["benchmark_label"] == "if7_trained_repository_linking_ranker"
    assert result["candidate_id"] == "IF7"
    assert result["training"]["candidate_examples"] > result["training"]["train_tasks"]
    assert result["models"]["task_aware_ranker"]["parameter_count"] >= 4
    assert "hebbian_pair_edge_score" in result["models"]["task_aware_ranker"][
        "feature_names"
    ]
    assert result["models"]["task_aware_ranker"]["loss_history"][0] > result["models"][
        "task_aware_ranker"
    ]["loss_history"][-1]
    assert "trained_task_aware_ranker" in result["methods"]
    assert "trained_no_hebbian_ranker" in result["methods"]
    assert "trained_ranker_beats_no_hebbian" in result
    assert result["methods"]["trained_task_aware_ranker"]["hit_at_k"] >= result[
        "methods"
    ]["raw_hebbian_context"]["hit_at_k"]
    assert result["best_method"]["name"] in result["methods"]
    assert result["sample_tasks"]


def test_if7_trained_repository_linking_ranker_cli_writes_machine_readable_record(
    tmp_path,
):
    corpus = tmp_path / "corpus.jsonl"
    _write_rows(corpus)
    output = tmp_path / "if7i.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_if7_repository_ranker.py",
            "--corpus-jsonl",
            str(corpus),
            "--train-split",
            "train",
            "--eval-split",
            "train",
            "--node-count",
            "512",
            "--max-memory-rows",
            "24",
            "--max-train-repositories",
            "4",
            "--max-eval-repositories",
            "4",
            "--negatives-per-query",
            "4",
            "--top-k",
            "2",
            "--epochs",
            "4",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.05",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--experiment-id",
            "IF7i_tmp",
            "--seed",
            "123",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "IF7i_tmp"
    assert record["status"] == "completed"
    assert record["metrics"]["benchmark_label"] == "if7_trained_repository_linking_ranker"
    assert "--epochs 4" in record["command"]
    assert (tmp_path / "manifest.json").exists()
