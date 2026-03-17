# Reproducibly build + serve CUDA-accelerated AI/ML stacks with [Flox](https://flox.dev) and [Nix](https://nixos.org/)

Flox is built on open-source [Nix](https://nixos.org), a reproducible package / environment manager and build system. Nix defines build recipes as code and treats each build as a pure function of its declared inputs. The same Nix expression works one month, one year, even five years after testing.

Flox supplements Nix with private catalogs; [FloxHub](https://hub.flox.dev) (a central site to share, version, and manage Flox environments); and declarative environment manifests expressed as TOML.

This repo hosts GPU- and CPU-targeted build recipes, production model-serving runtimes, model quantization tooling, monitoring environments, and other resources for the NVIDIA CUDA ecosystem. This repo would not exist without the dedicated excellence of the Nix ecosystem and the trailblazing ingenuity and brilliance of the Nix CUDA team. Thanks to both.

---

## What's here

### Build recipes

| Repository | What it does |
|---|---|
| **[pytorch-cuda](https://github.com/flox/pytorch-cuda)** | Parametric Nix builder for GPU- and CPU-targeted PyTorch. Covers PyTorch 2.8.0 through 2.10.0 on CUDA 12.8 through 13.1. Generates concrete, hardware-specific package definitions from metadata tables. |
| **[gpu-specific-pytorch-2.9.1](https://github.com/flox/gpu-specific-pytorch-2.9.1)** | Example repo to complement [this guide](https://flox.dev/blog/gpu-optimized-pytorch-builds-made-easy-with-flox-and-nix/). Custom PyTorch 2.9.1 variants with CUDA 12.9.1 support, targeted for specific GPU architectures (SM61 through SM120) and CPU instruction sets (AVX, AVX2, AVX-512, ARMv9). |
| **[onnx-runtime](https://github.com/flox/onnx-runtime)** | ONNX Runtime 1.18 through 1.24.2 for Python 3.12 and 3.13, CUDA 12.4 and 12.9. Versions segmented across Git branches. |
| **[magma](https://github.com/flox/magma)** | MAGMA 2.9.0 for NVIDIA GPUs. Single-architecture static builds that replace the ~10 GB all-architecture closure in nixpkgs. |
| **[vllm](https://github.com/flox/vllm)** | vLLM 0.13.0 through 0.15.1, with 0.16.0 coming soon. Versions segmented across Git branches. |
| **[llamacpp](https://github.com/flox/llamacpp)** | GPU-specific llama.cpp recipes pinned to specific versions, plus recipes that always build latest. |

### Model-serving runtimes

Each runtime is a declarative, reproducible Flox environment that runs directly on bare metal, in VMs, [uncontained on Kubernetes](https://flox.dev/blog/kubernetes-uncontained-explained-unlocking-faster-more-reproducible-deployments-on-k8s/), or as the basis for distroless OCI images. Containers aren't required, but (for container-based workflows) Flox and Nix make containers even better: minimal, truly declarative, deterministic.

| Repository | What it serves |
|---|---|
| **[triton-runtime](https://github.com/flox/triton-flox-runtime)** | NVIDIA Triton Inference Server v2.66.0 with Python, ONNX Runtime v1.24.2, TensorRT v10.23, and vLLM v0.15.1 backends. HTTP, gRPC, Prometheus metrics, and optional OpenAI-compatible frontend. |
| **[triton-runtime with tensorrt-llm](https://github.com/flox/triton-trtllm-flox-runtime)** | NVIDIA Triton Inference Server v2.66.0 with Python, ONNX Runtime v1.24.2, TensorRT v10.23, TensorRT-LLM 1.10, and vLLM v0.15.1 backends. HTTP, gRPC, Prometheus metrics, and optional OpenAI-compatible frontend. |
| **[triton-monitoring](https://github.com/flox/triton-monitoring-runtime)** | Example Grafana + Prometheus monitoring stack for NVIDIA Triton Inference Server v2.66.0. Just works everywhere: on x86-64 and ARM Linux and macOS; locally, in CI, in prod. |
| **[vllm-runtime](https://github.com/flox/vllm-flox-runtime)** | vLLM v0.16.0 on CUDA 12.9. OpenAI-compatible API, multi-GPU tensor and pipeline parallelism, multi-source model provisioning (local, HuggingFace, S3, R2), and three-stage model validation. |
| **[vllm-monitoring](https://github.com/flox/vllm-monitoring-runtime)** | Example Grafana + Prometheus monitoring stack for vLLM. Just works everywhere: on x86-64 and ARM Linux and macOS; locally, in CI, in prod. |  |
| **[llamacpp-runtime](https://github.com/flox/llamacpp-flox-runtime)** | llama.cpp on CUDA 12.9. Serves GGUF-quantized models via llama-server with continuous batching, Flash Attention, multi-GPU layer splitting, and GGUF artifact validation (magic bytes, header parsing, optional SHA256 pinning). |

### Quantization and conversion tooling

| Repository | What it does |
|---|---|
| **[model-quantizer](https://github.com/flox/model-quantizer)** | Quantize HuggingFace models for offline inference. AWQ 4-bit, FP8 via torchao, LLM Compressor (FP8), and GGUF for llama.cpp. Local and production command variants with strict validation, locking, and structured error reporting. x86-64 Linux. |
| **[triton-trtllm-tools](https://github.com/flox/triton-trtllm-tools)** | Convert HuggingFace models into TensorRT-LLM checkpoints, then compile checkpoints into TensorRT engines for Triton serving. Includes benchmarking, evaluation, pruning, refitting, and local validation tools. x86-64 Linux. |

---

## Why build GPU-specific packages?

Generic PyTorch wheels pull in support for more than half a dozen CUDA architectures, plus Intel- and Apple-specific backends. Building for the hardware you actually run:

- **Shrinks artifacts.** A targeted CUDA PyTorch closure is roughly 60% the size of upstream nixpkgs PyTorch. On macOS/Darwin, less than one-third: 2.66 GB. CPU-only builds clock in at just over 1.0 GB.
- **Improves performance.** Compiling for one SM architecture and/or one CPU ISA lets the compiler optimize without compromise.
- **Reduces the attack surface.** Fewer unused backends and code paths mean less is exposed at runtime.
- **Pins one artifact everywhere.** Publish targeted builds for dev, CI, and production instead of relying on whichever upstream packages happen to exist. Prototype and train on GPUs, optimize for GPU or CPU inferencing in eval, run minimal CPU- or GPU-optimized builds in production.

A final reason is that building with Nix is both rewarding and surprisingly straightforward. AI coding tools and agents are exceptionally fluent in Nix's functional language. These tools can reliably generate Nix expressions or flakes that reproduce exactly the same behavior and outcome one month, one year, or five years on.

---

## Why declarative environments for model serving?

Unlike a Dockerfile-first workflow, where the OCI image is the thing you build, tag, and promote, Flox and Nix make the declarative environment the unit of promotion.

The same Flox or Nix environment travels across the SDLC:

- On developers' CUDA-accelerated laptops and desktops
- On NVIDIA DGX Spark locally
- On Slurm-managed GPU clusters for research, eval, and batch inference
- On Kubernetes GPU clusters, VMs, or bare metal for eval and production

Changes are atomic edits committed to Git or published to FloxHub as a new generation. Rollbacks are switching to an earlier generation or, in GitOps flows, pointing to an earlier commit.

You can also build minimal, distroless containers from Flox or Nix environments.

---

## NVIDIA CUDA Kickstart Program

These repositories are part of the [Flox CUDA Kickstart Program](https://flox.dev/cuda-kickstart). Flox can help you customize and implement build recipes and serving environments for your own AI/ML CUDA workloads.

[@flox](https://github.com/flox) · [flox.dev](https://flox.dev) · [FloxHub](https://hub.flox.dev) · [LinkedIn](https://linkedin.com/company/floxdev) · [Bluesky](https://bsky.app/profile/flox.dev)
