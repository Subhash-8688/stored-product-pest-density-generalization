# Inspected Software and Hardware Environment

The following environment was inspected on June 1, 2026. Confirm that it matches the machine used for the final archived experiment runs before copying these values into a manuscript.

## Hardware

| Component | Inspected value |
| --- | --- |
| CPU | 20-core ARM processor: 10 Cortex-X925 cores and 10 Cortex-A725 cores |
| System memory | 121 GiB |
| GPU | NVIDIA GB10 |
| GPU memory | Unified-memory platform; dedicated GPU memory is reported as `Not Supported` by `nvidia-smi` |
| NVIDIA driver | 580.126.09 |

## Software

| Component | Inspected value |
| --- | --- |
| Operating system | Ubuntu 24.04.3 LTS |
| Python | 3.12.3 |
| PyTorch | 2.9.0+cu130 |
| CUDA toolkit reported by PyTorch | 13.0 |
| cuDNN | 9.13.0 |
| Ultralytics source version | 8.4.6 |
| Pinned Ultralytics commit | `eec4148e7b976cbbe1378aeee03f52337c79479e` |

The repository requirements and pinned source patch capture the software dependencies. For latency reporting, rerun timing measurements on the intended deployment hardware using repeated trials and a warm-up period.
