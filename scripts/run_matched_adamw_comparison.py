from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from weightlab.matched_comparison import (
    DEFAULT_COMPARISON_CONFIG,
    build_eval_commands,
    build_train_command,
    load_comparison_config,
    selected_runs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_COMPARISON_CONFIG)
    parser.add_argument("--run", action="append")
    parser.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_comparison_config(args.config)
    commands: list[list[str]] = []
    for run in selected_runs(config, args.run or ["all"]):
        if args.phase in {"train", "all"}:
            commands.append(build_train_command(config, run))
        if args.phase in {"eval", "all"}:
            commands.extend(build_eval_commands(config, run))

    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            if "--checkpoint" in command:
                checkpoint = Path(command[command.index("--checkpoint") + 1])
                if not checkpoint.exists():
                    print(f"skipping missing checkpoint: {checkpoint}", flush=True)
                    continue
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
