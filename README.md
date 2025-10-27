# DHSA: Dynamic Hierarchical Sparse Attention

This repository contains the official implementation of Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)


## Overview

LLMs face efficiency limits from the quadratic cost of dense attention. Static sparse methods (e.g., sliding windows, global tokens) reduce computation but cannot adapt to content, while dynamic ones still rely on heuristics. DHSA is a data-driven plug-in that predicts attention sparsity on the fly without retraining. It adaptively segments input into variable-length chunks, computes chunk-level similarities, and refines them into token-level importance scores for efficient long-context modeling.

<br>
<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DHSA/main/misc/Framework.png' width=650>
</p>
<br>

## Key Features

* **Dynamic Boundary Prediction** – Learns to segment input sequences into variable-length chunks based on semantic shifts.

* **Hierarchical Sparsity Prediction** – Combines chunk-level and token-level attention estimation for efficient long-context modelling.

* **Plug-and-Play Integration** – Works with existing Transformer layers without retraining base weights.

<br>
<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DHSA/main/misc/NIAH_gemma2.png' width=650>
</p>
<br>

Latency and Memory Comparison (Gemma2-2B, a single 24 GB GPU)

| Context Length | Method | Prefill Latency (s) | Prefill Peak Memory (GB) |
|----------------|---------|--------------|------------------|
| 8k  | Dense | 1.65 | 10.72 |
|          | **DHSA** | **1.19** | **6.91** |
| 16k | Dense | - | OOM |
|          | **DHSA** | **2.18** | **9.69** |
| 32k | Dense | - | OOM |
|          | **DHSA** | **4.51** | **16.99** |

*DHSA uses a 2k attention budget with a query chunk size of 256.*


## Repository Structure

```bash
DHSA/
├── boundary_predictor/
├── boundary_predictor_weights/
├── data/
├── results_longbench/
├── results_needle/
├── scripts/
├── utils/
├── eval_longbench.py
├── run_latency_test.py
├── run_longbench.py
├── run_needle_in_haystack.py
└── visualize_needle_in_haystack.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/xiongsiheng/DHSA.git
cd DHSA

# Create and activate a conda environment with Python 3.12
conda create -n dhsa python=3.12 -y
conda activate dhsa

# Install dependencies
pip install -r requirements.txt
```


## Quick Start

### Boundary Prediction

You can directly download the predictor weights [here](https://huggingface.co/sxiong/DHSA).

If you wish to train from scratch, you can download the following datasets: [Long Data Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections), [trivia QA](https://huggingface.co/datasets/mandarjoshi/trivia_qa), [ChatQA2](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data), and use the training scripts provided in `boundary_predictor/`.

### Needle-in-a-Haystack Test

To evaluate the model's ability to retrieve specific information from a long context, run:

```bash
bash scripts/run_needle_in_haystack.sh
```

To visualize the results, run:

```bash
bash scripts/run_visualize_needle_in_haystack.sh
```


### Latency & Memory Comparison

To benchmark latency and memory usage against other methods, run:

```bash
bash scripts/run_latency_test.sh
```


### LongBench

To test performance on the comprehensive [LongBench](https://github.com/THUDM/LongBench) suite, run:


```bash
bash scripts/run_longbench.sh
```



To evaluate the results, run:


```bash
bash scripts/run_eval_longbench_res.sh
```


## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.