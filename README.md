# llama.cpp Benchmarks on NVIDIA DGX Spark

This repository contains benchmarking scripts, model setup guides, and performance reports for running [llama.cpp](https://github.com/ggml-org/llama.cpp) on the **NVIDIA DGX Spark** (Grace Blackwell GB10).

The goal is to characterize the performance trade-offs of the GB10 architecture—specifically the interaction between its massive compute capability (approx. 1 PFLOP FP4) and its unified LPDDR5x memory bandwidth (~273 GB/s)—across Dense and Mixture-of-Experts (MoE) LLM architectures.

## Documentation & Reports

*   **[Performance Report](dgx_spark/report.md):** Detailed analysis of Prompt Processing (Prefill) and Decoding speeds.
    *   *Key Finding:* **MoE models are the "Killer App" for Spark.** The Qwen3 30B MoE reaches **~89 t/s**, while the similarly sized Qwen3 32B Dense hits a "bandwidth wall" at **~10.7 t/s**.
*   **[Installation Guide](dgx_spark/installation.md):** Step-by-step instructions for building `llama.cpp` with CUDA 13 on DGX Spark (fixing common `nvcc` path issues and configuring for Blackwell).
*   **[Model Download Guide](model_download.md):** Links and commands to retrieve the specific quantized GGUF models used in these benchmarks.

## Quick Start

### 1. Prerequisites
Ensure you have built `llama.cpp` using the [Installation Guide](dgx_spark/installation.md).

### 2. Setup Environment
Copy the `llama-bench` binary from your build folder to this repository's root:

### 3. Download Models
Create the directory and download the required GGUF files (Qwen3 and Ministral families):

```bash
mkdir -p ggufs
# Refer to model_download.md for the specific wget commands
```
*See [model_download.md](model_download.md) for the full list of download URLs.*

### 4. Run Benchmark
Execute the automated benchmarking script. This runs both Prompt Processing (Prefill) and Decoding tests across varying context lengths and batch sizes.

```bash
bash bench_ctx_and_decode.sh
```

## Repository Structure

```text
├── bench_ctx_and_decode.sh    # Main benchmarking automation script
├── model_download.md          # wget commands for specific GGUF models
├── llama-bench                # Binary (Place here after building)
├── ggufs/                     # Model storage directory
└── dgx_spark/
    ├── installation.md        # Build guide for CUDA 13 on Spark
    ├── XX.log                 # Detail benchmark logs
    └── report.md              # Performance analysis and results
```

##  Contributions

Contributions are welcome for more results and corresponding reports.

## Acknowledgements

1.*enchmarks based on `llama.cpp` (backend: CUDA via GGML).