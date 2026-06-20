import numpy as np

from weightlab.compression import compare_compression_methods


def test_compression_accounting_includes_metadata_and_reports_quality_tradeoff():
    matrix = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
    representative_matrix = np.arange(1024, dtype=np.float32).reshape(32, 32) / 1024.0

    rows = compare_compression_methods(matrix, rank=3)
    by_name = {row["method"]: row for row in rows}
    representative_rows = compare_compression_methods(representative_matrix, rank=4)
    representative_by_name = {row["method"]: row for row in representative_rows}

    assert by_name["fp32"]["total_bytes"] == matrix.nbytes
    assert by_name["int8_uniform"]["metadata_bytes"] > 0
    assert by_name["product_quantized_rows"]["metadata_bytes"] > 0
    assert by_name["low_rank_svd"]["total_bytes"] > 0
    assert (
        representative_by_name["product_quantized_rows"]["total_bytes"]
        < representative_by_name["fp32"]["total_bytes"]
    )
    assert by_name["product_quantized_rows"]["reconstruction_mse"] >= 0.0
    assert (
        by_name["low_rank_svd"]["reconstruction_mse"]
        < by_name["rank1_outer_product"]["reconstruction_mse"]
    )
    assert by_name["int4_uniform"]["total_bytes"] < by_name["fp32"]["total_bytes"]


def test_compression_includes_kronecker_and_tensor_train_baselines():
    matrix = np.arange(4096, dtype=np.float32).reshape(64, 64) / 4096.0

    rows = compare_compression_methods(matrix, rank=4)
    by_name = {row["method"]: row for row in rows}

    assert "kronecker_rank1" in by_name
    assert "tensor_train_4d" in by_name
    assert by_name["kronecker_rank1"]["metadata_bytes"] > 0
    assert by_name["tensor_train_4d"]["metadata_bytes"] > 0
    assert by_name["kronecker_rank1"]["total_bytes"] < by_name["fp32"]["total_bytes"]
    assert by_name["tensor_train_4d"]["total_bytes"] < by_name["fp32"]["total_bytes"]
    assert by_name["kronecker_rank1"]["reconstruction_mse"] >= 0.0
    assert by_name["tensor_train_4d"]["reconstruction_mse"] >= 0.0
