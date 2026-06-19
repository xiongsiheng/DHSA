CUDA_VISIBLE_DEVICES=0 python run_ruler.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --density 0.125 \
  --sparsity-mask DHSA_vs \
  --q-block-size 128 \
  --k-block-size 32 \
  --max_num_examples 10 \
  --eval_batch_size 1 \
  --report-latency \
  --save_dir results_ruler
