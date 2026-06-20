from weightlab.continual import (
    run_chronological_memory_experiment,
    run_trainable_adapter_vs_retrieval_experiment,
)


def test_chronological_experiment_tracks_retrieval_adapter_and_rollback():
    result = run_chronological_memory_experiment(seed=17)

    assert len(result["steps"]) >= 3
    assert result["final"]["retrieval_accuracy"] >= result["final"]["frozen_accuracy"]
    assert (
        result["final"]["adapter_with_replay_prior_accuracy"]
        >= result["final"]["adapter_no_replay_prior_accuracy"]
    )
    assert result["rollback"]["restored_version"] == 1
    assert result["rollback"]["accuracy_after_rollback"] == result["steps"][1]["retrieval_accuracy"]


def test_trainable_adapter_experiment_compares_weight_updates_with_retrieval():
    result = run_trainable_adapter_vs_retrieval_experiment(seed=19)

    assert len(result["steps"]) >= 4
    assert result["weight_update_count"] >= 2 * len(result["steps"])
    assert result["final"]["retrieval_accuracy"] >= result["final"][
        "adapter_with_replay_accuracy"
    ]
    assert result["final"]["adapter_with_replay_prior_accuracy"] > result["final"][
        "adapter_no_replay_prior_accuracy"
    ]
    assert result["final"]["adapter_with_replay_new_accuracy"] >= result["final"][
        "adapter_no_replay_new_accuracy"
    ]
    assert result["final"]["retrieval_storage_bytes"] < result["final"][
        "adapter_with_replay_storage_bytes"
    ]
    assert result["null_hypothesis_outcome"] in {
        "retrieval_not_beaten",
        "adapter_beats_retrieval",
    }
