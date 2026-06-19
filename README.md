# DHSA: Dynamic Hierarchical Sparse Attention

<p align="center">
  <img src="https://raw.githubusercontent.com/xiongsiheng/DHSA/main/misc/Performance_overview.png" width="800">
</p>

This repository contains the official implementation of **[ICML 26 Spotlight, NeurIPS 25 Efficient Reasoning Workshop]** paper [Long-Context Modeling with Dynamic Hierarchical Sparse Attention for Memory-Constrained LLM Inference](https://arxiv.org/pdf/2510.24606).

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)

## Overview

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DHSA/main/misc/Framework_v2.png' width=650>
</p>

LLMs face efficiency limits from the quadratic cost of dense attention. Static sparse methods (e.g., sliding windows, global tokens) reduce computation but cannot adapt to content. DHSA is a data-driven plugin that predicts attention sparsity on the fly without retraining the base model. By extending [Block-Sparse Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) to support variable-sized query and key blocks, DHSA better preserves accuracy under highly sparse regimes.

## Repository Structure

```bash
DHSA/
├── Block-Sparse-Attention/
├── data/
├── results_longbench/
├── results_NIAH/
├── results_ruler/
├── scripts/
├── utils/
├── eval_longbench.py
├── eval_ruler.py
├── measure_latency.py
├── run_longbench.py
├── run_needle_in_haystack.py
├── run_ruler.py
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

# Build Block-Sparse-Attention kernel
cd Block-Sparse-Attention

pip install packaging
pip install ninja

MAX_JOBS=2 NVCC_THREADS=1 \
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE \
BLOCK_SPARSE_ATTN_FORWARD_ONLY=TRUE \
BLOCK_SPARSE_ATTN_LLAMA8B_BF16_CAUSAL_ONLY=TRUE \
BLOCK_SPARSE_ATTN_CUDA_ARCHS=86 \
python setup.py build_ext --inplace

# Install dependencies
cd ..
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

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

```
@article{xiong2025long,
  title={Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs},
  author={Xiong, Siheng and Zou, Joe and Fekri, Faramarz and Cho, Yae Jee},
  journal={arXiv preprint arXiv:2510.24606},
  year={2025}
}
```