# Hardware

One machine for every benchmark here unless the benchmark's README says otherwise.

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB. PCIe Gen 4 x16 (the card supports Gen 5; `nvidia-smi` reports Gen 4 as this system's maximum).
- GPU power limit: 400 W, set at boot with `nvidia-smi -pl 400` (default 600 W). The GPU is idle apart from the benchmark, one config at a time.
- CPU: AMD Ryzen 9 7950X (16 cores, 32 threads).
- RAM: DDR5, 250 GiB usable.
- Board: Gigabyte B650 AORUS ELITE AX.
- Storage: Crucial P3 Plus 4 TB NVMe, ext4 on LUKS.
- OS: Ubuntu 24.04.2, kernel 6.8.0-147.
- NVIDIA driver 580.178.04 (CUDA 13.0), NVIDIA Container Toolkit 1.18.1.
- Rootless Docker 28.0.4, GPU through CDI (`BENCH_CDI=1`).
