from __future__ import annotations

import json
from pathlib import Path

REQUIRED_IDEA_FIELDS = {
    "id",
    "name",
    "source",
    "mechanism",
    "intended_benefit",
    "required_assumptions",
    "known_incompatibilities",
    "compatible_ideas",
    "current_evidence",
    "strongest_opposing_evidence",
    "smallest_falsifying_test",
    "status",
}

EXTERNAL_SOURCE_TYPES = {
    "paper",
    "official implementation",
    "benchmark report",
    "web claim",
}


def test_combination_matrix_tracks_required_idea_fields_and_contenders():
    markdown = Path("research/combination_matrix.md")
    payload_path = Path("results/combination_matrix.json")

    assert markdown.exists()
    assert payload_path.exists()

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    ideas = payload["ideas"]
    contenders = payload["contenders"]

    assert payload["schema_version"] == 1
    assert len(ideas) >= 8
    assert all(REQUIRED_IDEA_FIELDS <= set(idea) for idea in ideas)
    assert {idea["status"] for idea in ideas} <= {
        "active",
        "combine",
        "revise",
        "archive",
        "rejected",
    }
    assert {
        "conservative_local_coder",
        "modular_repository_learner",
        "novel_mechanism_contender",
    } <= {contender["id"] for contender in contenders}
    assert payload["stages"]["current_stage"] in {
        "stage_1_individual_tests",
        "stage_2_pairwise_combinations",
        "stage_3_contender_stacks",
        "stage_4_multi_idea_stacks",
        "stage_5_fractional_factorial_sweep",
    }


def test_external_combination_ideas_carry_primary_source_metadata():
    payload = json.loads(Path("results/combination_matrix.json").read_text(encoding="utf-8"))
    external_ideas = [
        idea
        for idea in payload["ideas"]
        if idea["source"] in EXTERNAL_SOURCE_TYPES
    ]

    assert len(external_ideas) >= 5
    assert any(idea["status"] in {"archive", "rejected"} for idea in external_ideas)

    for idea in external_ideas:
        sources = idea.get("primary_sources")
        assert sources, idea["id"]
        for source in sources:
            assert {"title", "url", "access_date"} <= set(source), idea["id"]
            assert source["url"].startswith("https://"), idea["id"]
