from pathlib import Path

from weightlab.matched_comparison import (
    build_eval_commands,
    build_train_command,
    load_comparison_config,
    selected_runs,
)


def test_canonical_matched_config_builds_dense_and_adapter_commands():
    config = load_comparison_config(Path("configs/d4_adamw_fp32_matched_comparison.json"))
    runs = selected_runs(config, ["A_dense_544", "B_adapter_528", "C_dense_528"])

    dense_544 = build_train_command(config, runs[0])
    adapter_528 = build_train_command(config, runs[1])
    dense_528 = build_train_command(config, runs[2])

    assert "--corpus-jsonl" in dense_544
    assert "data/hf_mirror/exploratory_d3/corpus.jsonl" in dense_544
    assert dense_544[dense_544.index("--hidden-dim") + 1] == "544"
    assert adapter_528[adapter_528.index("--hidden-dim") + 1] == "528"
    assert dense_528[dense_528.index("--hidden-dim") + 1] == "528"
    assert adapter_528[adapter_528.index("--architecture-variant") + 1] == "adapter"
    assert adapter_528[adapter_528.index("--adapter-dim") + 1] == "64"
    for command in [dense_544, adapter_528, dense_528]:
        assert command[command.index("--optimizer-name") + 1] == "adamw"
        assert command[command.index("--mixed-precision") + 1] == "fp32"
        assert command[command.index("--attention-mask-mode") + 1] == "finite_causal"
        assert command[command.index("--block-impl") + 1] == "explicit_causal"
        assert command[command.index("--learning-rate") + 1] == "0.0001"
        assert command[command.index("--steps") + 1] == "10000"
        assert command[command.index("--validation-batches") + 1] == "512"
        assert command[command.index("--seed") + 1] == "123"
        assert command[command.index("--validation-seed") + 1] == "424242"
        assert command[command.index("--max-documents") + 1] == "0"


def test_canonical_matched_config_builds_final_and_best_eval_commands():
    config = load_comparison_config(Path("configs/d4_adamw_fp32_matched_comparison.json"))
    run = selected_runs(config, ["B_adapter_528"])[0]

    commands = build_eval_commands(config, run)

    assert len(commands) == 4
    assert {command[command.index("--split") + 1] for command in commands} == {
        "validation",
        "test",
    }
    assert any(
        any(part.endswith("dense_decoder_last.pt") for part in command) for command in commands
    )
    assert any(
        any(part.endswith("dense_decoder_best.pt") for part in command) for command in commands
    )
    for command in commands:
        assert command[command.index("--batches") + 1] == "512"
        if command[command.index("--split") + 1] == "validation":
            assert command[command.index("--seed") + 1] == "424242"
        if command[command.index("--split") + 1] == "test":
            assert command[command.index("--seed") + 1] == "424243"
