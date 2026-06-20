# Environment

Access date: 2026-06-20.

## Recorded System Summary

- OS: Linux x86_64, CachyOS.
- Kernel family: Linux 7.1 release-candidate series.
- Python: system Python 3.14.5; `uv` selected CPython 3.12.13 for the project environment.
- Git: 2.54.0.
- Compilers: GCC 16.1.1, Clang 22.1.6.
- RAM: about 30 GiB total.
- Disk at repository filesystem: about 314 GiB available before experiment outputs.
- Accelerator: AMD Radeon RX 9070 XT visible through ROCm SMI, about 16 GiB VRAM.
- Project PyTorch runtime after ROCm implementation: `torch 2.12.1+rocm7.2`, `torch.version.hip=7.2.53211`, `torch.cuda.is_available()=True`, device 0 reports `AMD Radeon RX 9070 XT`, and a ROCm-backed Torch matmul smoke check succeeded.
- Python libraries observed before project sync: PyTorch and NumPy installed globally; pytest, scikit-learn, datasets, faiss, and annoy missing globally.
- Internet: available during bootstrap.
- Git: repository already initialized and had no commits.

Private hostnames, usernames, GUIDs, and unnecessary device identifiers are intentionally omitted.

## Resource Assumptions

Initial experiments must run on CPU, avoid silent downloads, and remain small enough for repeated smoke tests. The initial `uv` environment installed PyTorch wheels; no model weights or datasets were downloaded.
