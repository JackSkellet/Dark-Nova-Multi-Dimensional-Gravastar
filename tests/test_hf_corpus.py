from weightlab.hf_corpus import HFSourceSpec, build_hf_corpus_manifest


def test_hf_corpus_manifest_records_mixed_license_configs_and_revision():
    def fetch_json(url: str):
        if "/api/datasets/example/code" in url:
            return {
                "sha": "abc123",
                "lastModified": "2026-01-01T00:00:00.000Z",
                "tags": ["license:mit"],
                "cardData": {"license": "mit"},
            }
        if "/size?dataset=example/code" in url:
            return {
                "size": {
                    "splits": [
                        {
                            "config": "Python-mit",
                            "split": "train",
                            "num_rows": 10,
                            "num_bytes_parquet_files": 1000,
                            "num_bytes_memory": 2000,
                        },
                        {
                            "config": "Python-gpl-3.0",
                            "split": "train",
                            "num_rows": 20,
                            "num_bytes_parquet_files": 3000,
                            "num_bytes_memory": 5000,
                        },
                    ]
                }
            }
        if "/parquet?dataset=example/code" in url:
            return {
                "parquet_files": [
                    {
                        "config": "Python-mit",
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/0000.parquet",
                        "size": 1000,
                    },
                    {
                        "config": "Python-gpl-3.0",
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/gpl.parquet",
                        "size": 3000,
                    },
                ]
            }
        raise AssertionError(url)

    manifest = build_hf_corpus_manifest(
        [
            HFSourceSpec(
                dataset="example/code",
                revision="abc123",
                configs=["Python-mit", "Python-gpl-3.0"],
                split="train",
                card_review="unit-test card review",
            )
        ],
        fetch_json=fetch_json,
    )

    assert manifest["source_count"] == 1
    assert manifest["accepted_config_count"] == 2
    assert manifest["rejected_config_count"] == 0
    assert manifest["total_rows"] == 30
    assert manifest["total_parquet_bytes"] == 4000
    assert manifest["filter_policy"]["corpus_use"] == "exploratory-research-only"
    source = manifest["sources"][0]
    assert source["resolved_revision"] == "abc123"
    assert source["accepted_configs"][0]["config"] == "Python-mit"
    assert source["accepted_configs"][0]["license"] == "mit"
    assert source["accepted_configs"][1]["config"] == "Python-gpl-3.0"
    assert source["accepted_configs"][1]["license"] == "gpl-3.0"
    assert source["accepted_configs"][1]["corpus_use"] == "exploratory-research-only"
    assert source["accepted_configs"][0]["parquet_files"][0]["sha256"] == (
        "metadata_only_not_downloaded"
    )


def test_hf_corpus_manifest_accepts_all_language_permissive_config():
    def fetch_json(url: str):
        if "/api/datasets/example/code" in url:
            return {"sha": "abc123", "cardData": {"license": "mit"}, "tags": ["license:mit"]}
        if "/size?dataset=example/code" in url:
            return {
                "size": {
                    "configs": [
                        {
                            "config": "all-mit",
                            "num_rows": 233000,
                            "num_bytes_parquet_files": 1844197149,
                            "num_bytes_memory": 5021216929,
                        },
                        {
                            "config": "all",
                            "num_rows": 999,
                            "num_bytes_parquet_files": 999,
                        },
                    ]
                }
            }
        if "/parquet?dataset=example/code" in url:
            return {
                "parquet_files": [
                    {
                        "config": "all-mit",
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/all-mit.parquet",
                        "size": 100,
                    },
                    {
                        "config": "all",
                        "split": "train",
                        "filename": "0000.parquet",
                        "url": "https://example.test/all.parquet",
                        "size": 999,
                    }
                ]
            }
        raise AssertionError(url)

    manifest = build_hf_corpus_manifest(
        [
            HFSourceSpec(
                dataset="example/code",
                revision="abc123",
                configs=["all-mit", "all"],
                split="train",
            )
        ],
        fetch_json=fetch_json,
    )

    assert manifest["accepted_config_count"] == 2
    assert manifest["rejected_config_count"] == 0
    assert manifest["sources"][0]["accepted_configs"][0]["language"] == "all"
    assert manifest["sources"][0]["accepted_configs"][0]["license"] == "mit"
    assert manifest["sources"][0]["accepted_configs"][1]["license"] == "unknown"
    assert manifest["sources"][0]["accepted_configs"][1]["license_metadata_status"] == (
        "recorded_mixed_or_unclear_exploratory_only"
    )
