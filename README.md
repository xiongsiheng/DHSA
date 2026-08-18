# DHSA: Dynamic Hierarchical Sparse Attention

This repository contains the official implementation of **[ICML 26 Spotlight, NeurIPS 25 Efficient Reasoning Workshop]** paper [Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference](https://arxiv.org/pdf/2510.24606).

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Predictor Checkpoints](#predictor-checkpoints)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)

## Overview

LLMs face efficiency limits from the quadratic cost of dense attention. 
Static sparse methods (e.g., sliding windows, global tokens) reduce computation but cannot adapt to content. 
DHSA is a data-driven plugin that predicts attention sparsity on the fly without retraining the base model. 
By extending [Block-Sparse Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) with **variable-sized query and key blocks** and **effective sparsity prediction**, DHSA better preserves accuracy under highly sparse regimes.

This release supports Qwen2/Qwen2.5 and Llama with the following sparse attention mask:

| Method | Description | Checkpoint |
| --- | --- | --- |
| `DHSA_learned_topK_static` | Learned TopK with a fixed row budget | Yes |
| `DHSA_learned_topK_dynamic` | Learned TopK with query-dependent row budgets | Yes |
| `DHSA_vs_optimized` | Optimized vertical-slash blockwise mask | No |
| `DHSA_vsb_memory_efficient` | Vertical-slash blockwise mask for memory-constrained long contexts | No |

DHSA handles prompt prefill. 
Autoregressive decoding falls back to the model's original [FlashAttention-2](https://github.com/dao-ailab/flash-attention) implementation.

## Results

NIAH uses the repository runner's score and is not on the same scale as [LongBench](https://github.com/THUDM/LongBench) or [RULER](https://github.com/NVIDIA/RULER). 
Compare values vertically within each column.

#### Llama-3.1-8B-Instruct BF16

| Mask                 | Density |    NIAH | LongBench |    RULER |
| -------------------- | ------: | ------: | --------: | -------: |
| Full FA2             |    100% |     6.2 |      39.5 |     86.4 |
| VS optimized         |   6.25% |     6.6 |      42.2 |     86.0 |
| VS optimized         |   12.5% |     6.6 |  **47.4** | **87.8** |
| Learned topK static  |   6.25% | **7.5** |      26.9 |     66.7 |
| Learned topK static  |   12.5% |     6.3 |      42.3 |     75.8 |
| Learned topK dynamic |   6.25% |     7.2 |      30.7 |     62.7 |
| Learned topK dynamic |   12.5% |     6.4 |      40.4 |     75.8 |

#### Llama-3.1-8B-Instruct 4-bit

| Mask                 | Density |    NIAH | LongBench |    RULER |
| -------------------- | ------: | ------: | --------: | -------: |
| Full FA2             |    100% |     7.7 |      35.7 |     89.3 |
| VS optimized         |   6.25% |     7.3 |      36.7 |     79.3 |
| VS optimized         |   12.5% |     7.7 |      36.6 | **88.7** |
| Learned topK static  |   6.25% |     7.4 |  **37.7** |     51.8 |
| Learned topK static  |   12.5% |     7.3 |      36.4 |     62.0 |
| Learned topK dynamic |   6.25% | **7.8** |      37.4 |     52.4 |
| Learned topK dynamic |   12.5% |     7.7 |      36.3 |     68.7 |

#### Qwen2.5-3B-Instruct BF16

| Mask                 | Density |    NIAH | LongBench |    RULER |
| -------------------- | ------: | ------: | --------: | -------: |
| Full FA2             |    100% |     8.2 |      43.6 |     56.7 |
| VS optimized         |   6.25% |     8.3 |  **40.2** | **61.8** |
| VS optimized         |   12.5% |     8.0 |      38.9 |     61.6 |
| Learned topK static  |   6.25% |     7.9 |      32.5 |     36.7 |
| Learned topK static  |   12.5% | **8.8** |      35.3 |     42.7 |
| Learned topK dynamic |   6.25% |     8.1 |      24.2 |     27.3 |
| Learned topK dynamic |   12.5% | **8.8** |      37.2 |     35.1 |

### Kernel-level Latency

Latency is measured on one A100 40 GB GPU for a single attention layer, including mask construction and the attention kernel. 
Values are median milliseconds from five measured iterations after two warmups. 
Parentheses show speedup over Full FA2 measured in the same run.
We do not distinguish between BF16 and 4-bit settings because quantization does not affect kernel-level latency.

#### Llama-3.1-8B-Instruct

| Mask                 | Density |        32K (ms) |        64K (ms) |        128K (ms) |
| -------------------- | ------: | --------------: | --------------: | ---------------: |
| Full FA2             |    100% |            40.3 |           164.3 |            676.4 |
| VS optimized         |   6.25% |     19.0 (2.1x) |     60.2 (2.7x) |     214.9 (3.2x) |
| VS optimized         |   12.5% |     28.5 (1.4x) |     95.6 (1.7x) |     369.0 (1.8x) |
| Learned topK static  |   6.25% | **14.8 (2.7x)** | **51.3 (3.2x)** |     198.4 (3.4x) |
| Learned topK static  |   12.5% |     19.9 (2.0x) |     72.2 (2.3x) |     281.4 (2.4x) |
| Learned topK dynamic |   6.25% |     16.4 (2.5x) |     52.4 (3.1x) | **197.7 (3.4x)** |
| Learned topK dynamic |   12.5% |     21.7 (1.9x) |     74.0 (2.2x) |     283.5 (2.4x) |

#### Qwen2.5-3B-Instruct

| Mask                 | Density |       32K (ms) |        64K (ms) |       128K (ms) |
| -------------------- | ------: | -------------: | --------------: | --------------: |
| Full FA2             |    100% |           20.5 |            82.3 |           337.9 |
| VS optimized         |   6.25% |    10.2 (2.0x) |     30.6 (2.7x) |    109.9 (3.1x) |
| VS optimized         |   12.5% |    14.8 (1.4x) |     50.7 (1.6x) |    184.0 (1.8x) |
| Learned topK static  |   6.25% | **8.7 (2.4x)** | **26.6 (3.1x)** |     99.9 (3.4x) |
| Learned topK static  |   12.5% |    11.3 (1.8x) |     37.1 (2.2x) |    142.2 (2.4x) |
| Learned topK dynamic |   6.25% |    10.3 (2.0x) |     28.2 (2.9x) | **99.8 (3.4x)** |
| Learned topK dynamic |   12.5% |    12.9 (1.6x) |     40.0 (2.1x) |    143.8 (2.3x) |

## Repository Structure

```text
DHSA/
├── Block-Sparse-Attention/
├── checkpoints/
├── data/
├── results_latency/
├── results_longbench/
├── results_NIAH/
├── results_ruler/
├── scripts/
├── tests/
├── utils/
├── eval_longbench.py
├── eval_ruler.py
├── measure_latency.py
├── run_longbench.py
├── run_needle_in_haystack.py
├── run_ruler.py
├── topk_predictor.py
└── visualize_needle_in_haystack.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/xiongsiheng/DHSA.git
cd DHSA

# Create and activate a conda environment with Python 3.12
conda create -n DHSA python=3.12 -y
conda activate DHSA

pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Build the Block-Sparse-Attention kernel
bash scripts/build_kernel.sh
```

Our default build supports A100 (`sm_80`), RTX 3090 (`sm_86`), and RTX 4090 (`sm_89`). 
Override the architecture list for a machine-specific build:

```bash
BLOCK_SPARSE_ATTN_CUDA_ARCHS=86 bash scripts/build_kernel.sh
```

The sparse kernel is BF16, causal, forward-only, and requires head dimension 128. 
The validated Transformers version is `4.45.0`.

## Predictor Checkpoints

The released predictor weights are provided in the [DHSA collection](https://huggingface.co/collections/sxiong/dhsa).

| Model mode | HF repository | Static checkpoint | Dynamic checkpoint |
| --- | --- | --- | --- |
| Qwen2.5-3B BF16 | [sxiong/DHSA-Qwen2.5-3B-Instruct-BF16](https://huggingface.co/sxiong/DHSA-Qwen2.5-3B-Instruct-BF16) | `predictor_static.pt` | `predictor_dynamic.pt` |
| Llama-3.1-8B BF16 | [sxiong/DHSA-Llama-3.1-8B-Instruct-BF16](https://huggingface.co/sxiong/DHSA-Llama-3.1-8B-Instruct-BF16) | `predictor_static.pt` | `predictor_dynamic.pt` |
| Llama-3.1-8B NF4 | [sxiong/DHSA-Llama-3.1-8B-Instruct-NF4](https://huggingface.co/sxiong/DHSA-Llama-3.1-8B-Instruct-NF4) | `predictor_static.pt` | `predictor_dynamic.pt` |

Download the repositories into `checkpoints/`:

```bash
hf download sxiong/DHSA-Qwen2.5-3B-Instruct-BF16 \
  --local-dir checkpoints/DHSA-Qwen2.5-3B-Instruct-BF16

hf download sxiong/DHSA-Llama-3.1-8B-Instruct-BF16 \
  --local-dir checkpoints/DHSA-Llama-3.1-8B-Instruct-BF16

hf download sxiong/DHSA-Llama-3.1-8B-Instruct-NF4 \
  --local-dir checkpoints/DHSA-Llama-3.1-8B-Instruct-NF4
```

The downloaded directory structure is:

```text
checkpoints/
├── DHSA-Qwen2.5-3B-Instruct-BF16/
│   ├── predictor_static.pt
│   └── predictor_dynamic.pt
├── DHSA-Llama-3.1-8B-Instruct-BF16/
│   ├── predictor_static.pt
│   └── predictor_dynamic.pt
└── DHSA-Llama-3.1-8B-Instruct-NF4/
    ├── predictor_static.pt
    └── predictor_dynamic.pt
```

All predictor checkpoints support both `6.25%` and `12.5%` density. Static checkpoints use a fixed row budget, while dynamic checkpoints use query-dependent row budgets and include density-specific budget configurations.

## Quick Start

### Needle-in-a-Haystack Test

To evaluate the model's ability to retrieve specific information from a long context, run:

```bash
bash scripts/run_needle_in_haystack_llama3.sh

bash scripts/run_needle_in_haystack_llama3_100k.sh

bash scripts/run_needle_in_haystack_qwen2_5.sh
```

To visualize the results, run:

```bash
bash scripts/visualize_NIAH_res.sh
```

### LongBench

To test performance on the comprehensive [LongBench](https://github.com/THUDM/LongBench) suite, run:

```bash
bash scripts/run_longbench_llama3.sh

bash scripts/run_longbench_qwen2_5.sh
```

To evaluate the results, run:

```bash
bash scripts/eval_longbench.sh
```

### RULER

To evaluate performance on the controlled synthetic [RULER](https://github.com/NVIDIA/RULER) suite, you can either generate the data using the official repository or download our pre-generated [data](https://huggingface.co/datasets/sxiong/DHSA_RULER), then run:

```bash
bash scripts/run_ruler_llama3.sh

bash scripts/run_ruler_qwen2_5.sh
```

To evaluate the results, run:

```bash
bash scripts/eval_ruler.sh
```

### Latency Measurement

To benchmark latency and memory usage against [Flash Attention2](https://github.com/dao-ailab/flash-attention), run:

```bash
bash scripts/measure_latency.sh

bash scripts/measure_latency_batched.sh
```

## Acknowledgements

The implementation is built upon [Block-Sparse Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) and [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory).
We sincerely appreciate these teams for their open-source contributions.

## Citation

```bibtex
@inproceedings{xionglong,
  title={Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference},
  author={Xiong, Siheng and Zou, Joe and Fekri, Faramarz and Cho, Yae Jee},
  booktitle={Forty-third International Conference on Machine Learning}
}
```
