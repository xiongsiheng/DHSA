CUDA_VISIBLE_DEVICES=0 python measure_latency.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --sparsity_mask DHSA_topk \
  --q-block-size 128 \
  --k-block-size 32 \
  --eval_batch_size 8