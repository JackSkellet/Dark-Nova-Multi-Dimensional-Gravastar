.PHONY: test lint experiments public-chronology status

test:
	uv run pytest

lint:
	uv run ruff check .

experiments:
	uv run python scripts/run_experiments.py --seed 123

public-chronology:
	uv run python scripts/run_public_repo_chronology.py --repo-path $(REPO) --max-commits 12 --top-k 5

status:
	sed -n '1,220p' research/STATUS.md
