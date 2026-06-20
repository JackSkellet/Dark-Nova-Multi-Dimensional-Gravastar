# Lookup Results

Status: generated from `results/E4_component_lookup.json`, `results/E4b_routed_execution.json`, and `results/E4c_torch_batched_routed_execution.json`.

The first benchmark measures isolated CPU lookup. It does not yet include authorization filtering, GPU transfer, decompression, dispatch, or kernel execution, so it cannot support an end-to-end speed claim.

## Seed 123 Results

At bank size 16, dense top-k p50 was 0.0023 ms with perfect recall. Tree routing was slower at 0.0060 ms and recall 0.902. Hash routing was slower at 0.0092 ms and recall 0.980.

At bank size 128, dense top-k p50 was 0.0036 ms with perfect recall. Tree routing was 0.0064 ms with recall 0.641. Hash routing was 0.0113 ms with recall 0.785.

At bank size 1024, exact vector search p50 was 0.0144 ms and dense top-k p50 was 0.0172 ms, both with perfect recall. Tree routing was faster at 0.0076 ms but recall fell to 0.180. Hash routing used less index memory but p50 was 0.0242 ms with recall 0.645.

## Interpretation

No indexed method produced an end-to-end win at acceptable recall in this reduced CPU setting. Dense/exact routing remains the stronger baseline for small-to-medium banks. H4 needs larger banks, realistic batching, authorization filtering, cache reuse, component transfer, and dispatch before it can be reconsidered.

## E4b: Routed Execution Simulation

Seed 123 follow-up with authorization filtering, repeated queries, transfer, reconstruction, and dispatch timing:

| Bank | Method | Recall | Cache hit | End-to-end p50 ms | End-to-end p95 ms | Lookup p50 ms | Index bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | Dense router | 1.000 | 0.000 | 0.0134 | 0.0149 | 0.0058 | 16384 |
| 128 | Exact index | 1.000 | 0.000 | 0.0095 | 0.0130 | 0.0021 | 16384 |
| 128 | Cached exact | 1.000 | 0.500 | 0.0078 | 0.0129 | 0.0010 | 16384 |
| 128 | Hash index | 0.948 | 0.000 | 0.0201 | 0.0220 | 0.0128 | 3072 |
| 2048 | Dense router | 1.000 | 0.000 | 0.0239 | 0.0251 | 0.0185 | 262144 |
| 2048 | Exact index | 1.000 | 0.000 | 0.0087 | 0.0089 | 0.0035 | 262144 |
| 2048 | Cached exact | 1.000 | 0.500 | 0.0065 | 0.0096 | 0.0017 | 262144 |
| 2048 | Hash index | 0.719 | 0.000 | 0.0502 | 0.0515 | 0.0449 | 26112 |
| 8192 | Dense router | 1.000 | 0.000 | 0.1122 | 0.1146 | 0.1029 | 1048576 |
| 8192 | Exact index | 1.000 | 0.000 | 0.0237 | 0.0242 | 0.0158 | 1048576 |
| 8192 | Cached exact | 1.000 | 0.500 | 0.0141 | 0.0246 | 0.0076 | 1048576 |
| 8192 | Hash index | 0.708 | 0.000 | 0.1569 | 0.1592 | 0.1514 | 99840 |

Interpretation: in this CPU simulation, exact indexing starts to matter at larger banks and repeated-query caching provides a measured end-to-end benefit. Hash indexing saves index memory but loses too much recall and is slower in this implementation. This is still a simulation: transfer and reconstruction are CPU copies/math, not real GPU transfer, decompression kernels, or batched model execution.

## E4c: Torch Batched Routed Execution

Seed 123 follow-up with real Torch tensor dispatch. The benchmark requests `device=rocm` and reports a logical `accelerator_backend` separately from PyTorch's internal device type. The project environment now uses PyTorch `2.12.1+rocm7.2` with HIP runtime `7.2.53211`; PyTorch still exposes the execution device type as `cuda`, which is expected for ROCm builds. This run reports `rocm_available=1` and `uses_rocm_transfer=1`, so E4c now exercises real ROCm-backed tensor movement and dispatch.

| Bank | Requested | Backend | Device | ROCm available | Recall | Batch | End-to-end p50 ms | End-to-end p95 ms | Lookup p50 ms | Transfer p50 ms | Dispatch p50 ms | Throughput q/s |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | rocm | rocm | cuda | 1 | 1.000 | 32 | 0.5172 | 30.1241 | 0.0471 | 0.2137 | 0.0693 | 3480.5 |
| 2048 | rocm | rocm | cuda | 1 | 1.000 | 32 | 0.3888 | 0.6132 | 0.0369 | 0.1641 | 0.0634 | 70848.3 |

Interpretation: E4c replaces part of the earlier hand-written CPU simulation with a real ROCm-backed Torch execution path. The small benchmark is still not enough to claim a speed win: bank 512 shows a high p95 outlier, and the test does not measure occupancy, kernel fusion, decompression kernels, or realistic model layers. It does, however, remove the previous CPU-fallback blocker.

## E4d: ROCm Transfer Scaling

Seed 123 follow-up with ROCm-requested transfer scaling. This benchmark isolates host-to-device transfer, one simple device dispatch operation, and device-to-host transfer for three payload sizes. It uses PyTorch ROCm/HIP and records that it does not measure occupancy, power, kernel fusion, decompression kernels, or useful model layers.

| Payload MiB | Backend | ROCm available | H2D p50 ms | H2D GB/s | Dispatch p50 ms | D2H p50 ms | D2H GB/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | rocm | 1 | 0.0865 | 12.122 | 0.0487 | 0.0802 | 13.076 |
| 8 | rocm | 1 | 0.3660 | 22.917 | 0.0572 | 0.2921 | 28.715 |
| 32 | rocm | 1 | 1.9330 | 17.359 | 0.1508 | 1.8368 | 18.268 |

Interpretation: E4d provides a cleaner ROCm transfer baseline than E4c's routed-dispatch timings. It is measurement-positive because ROCm transfer scaling is recorded across payload sizes, but it still does not show an architecture-level speed advantage.
