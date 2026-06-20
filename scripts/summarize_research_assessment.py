from __future__ import annotations

import argparse
import json
from pathlib import Path

from weightlab.assessment import assess_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize limitation-aware research outcomes from results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/research_assessment.json"),
    )
    args = parser.parse_args()

    summary = assess_manifest(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "wrote"
        f" {args.output}"
        f" records={summary['record_count']}"
        f" pareto_improvement_found={summary['pareto_improvement_found']}"
        f" pareto_dominance_found={summary['pareto_dominance_found']}"
        f" frontier_expansion_found={summary['frontier_expansion_found']}"
    )


if __name__ == "__main__":
    main()
