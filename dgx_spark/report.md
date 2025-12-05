# llama.cpp Performance on NVIDIA DGX Spark

## 1. Brief Motivation
The NVIDIA DGX Spark presents a unique architectural profile for local LLM inference:
*   **High Compute:** NVIDIA GB10 Grace Blackwell superchip (approx. 1 PFLOP FP4).
*   **Low Bandwidth:** 128 GB LPDDR5x unified memory (~273 GB/s), significantly slower than HBM-based server GPUs.

This benchmark characterizes this trade-off. We expect strong prompt processing (compute-bound) but potential bottlenecks in decoding (memory-bandwidth-bound), specifically testing where the "bandwidth wall" hits across different model sizes.

## 2. Model Selection Motivation
We selected **Qwen 3** and **Ministral 3** families in **Q4_K_M** quantization to cover a wide range of parameter counts and architectures:

1.  **Small (1.7B - 3B):** To test raw kernel overhead and maximum theoretical throughput.
2.  **Medium (8B - 14B):** The sweet spot for local assistants; testing if bandwidth limits start to appear.
3.  **Large Dense (32B):** To stress-test the memory bandwidth limit.
4.  **Large MoE (30B A3B):** To see if Mixture-of-Experts architectures can bypass the bandwidth limitations of dense models on this hardware.

## 3. Results

### 3.1 Prompt Processing (Prefill)
*Measured in Tokens per Second (t/s).*

| Model Family | Model | Size | PP Speed (512 ctx) | PP Speed (16k ctx) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3** | 1.7B | 1.72 B | **11,947 t/s** | 5,208 t/s |
| | 8B | 8.19 B | 3,167 t/s | 1,862 t/s |
| | **30B (MoE)** | 30.53 B | 2,541 t/s | 2,059 t/s |
| | **32B (Dense)**| 32.76 B | 762 t/s | 481 t/s |
| **Ministral**| 3B | 3.43 B | 6,961 t/s | 3,166 t/s |
| | 8B | 8.49 B | 3,020 t/s | 1,868 t/s |
| | 14B | 13.51 B | 1,853 t/s | 1,284 t/s |

### 3.2 Decoding (Generation)
*Measured in Tokens per Second (t/s). Batch size = 1. Comparisons at shallow (512) vs deep (2048) context.*

| Model Family | Model | Size | Decode (512 ctx) | Decode (2048 ctx) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3** | 1.7B | 1.72 B | **161.4 t/s** | 146.1 t/s |
| | 8B | 8.19 B | 43.7 t/s | 42.0 t/s |
| | **30B (MoE)** | 30.53 B | **89.3 t/s** | **83.8 t/s** |
| | **32B (Dense)**| 32.76 B | 10.7 t/s | 10.5 t/s |
| **Ministral**| 3B | 3.43 B | 91.9 t/s | 86.6 t/s |
| | 8B | 8.49 B | 41.7 t/s | 40.1 t/s |
| | 14B | 13.51 B | 26.4 t/s | 25.7 t/s |

## 4. Brief Explanation

1.  **The Bandwidth Wall (Dense Models):**
    The difference between the **32B Dense** (~10 t/s) and **30B MoE** (~89 t/s) is the defining characteristic of the DGX Spark.
    *   The **32B Dense** model requires reading ~18GB of data per token generated. With ~273 GB/s bandwidth, the theoretical max is ~15 t/s. We achieved ~10.7 t/s, confirming the system is hard-bound by LPDDR5x memory bandwidth.
    *   The **8B Dense** models hover around ~43 t/s, which is usable but significantly slower than desktop GPUs with GDDR6X or server GPUs with HBM.

2.  **MoE is the "Killer App" for Spark:**
    The **Qwen3 30B MoE** performs exceptionally well (~89 t/s). Because it only activates a small subset of parameters (approx 2.4B) per token, it avoids the memory bandwidth bottleneck that cripples the dense 32B model, while still providing 30B-class model capacity in memory.

3.  **Compute Headroom (Prefill):**
    Prompt processing speeds are excellent (up to 11k t/s). This confirms the GB10 chip has massive compute power (Tensor Cores). Long-context ingestion is efficient, only dropping significantly at extreme lengths (16k) or for the heavy 32B dense model.

**Conclusion:** The DGX Spark is an excellent inference machine for **Mixture-of-Experts (MoE)** models and smaller dense models (<14B). However, for large dense models (>30B), the LPDDR5x memory bandwidth creates a severe bottleneck, reducing generation speed to ~10 t/s.