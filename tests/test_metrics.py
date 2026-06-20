import subprocess

from weightlab import metrics
from weightlab.metrics import ExperimentRecord


def test_experiment_record_marks_dirty_worktree(monkeypatch):
    def fake_check_output(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "abc123\n"
        if command == ["git", "status", "--short"]:
            return " M src/example.py\n?? results/example.json\n"
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(metrics.subprocess, "check_output", fake_check_output)

    record = ExperimentRecord(
        experiment_id="dirty_record",
        hypothesis="metadata",
        seed=123,
        command="pytest",
        metrics={},
    ).to_jsonable()

    assert record["git_commit"] == "abc123"
    assert record["git_dirty"] is True
    assert record["git_status_short"] == "M src/example.py\n?? results/example.json"
