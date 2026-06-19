"""
Base model utilities for boundary predictor training.
"""
import torch
import transformers

from torch.nn.functional import pad

# Disable torch dynamo for stability
torch._dynamo.config.disable = True


from utils.config import *
from utils.monkeypatch import replace_gemma, _configure_attention_layers



def load_labels(model, label_data):
    """Load labels from file into model to save computation time."""
    layers = len(model.model.layers)
    for i in range(layers):
        model.model.layers[i].self_attn.ratios = label_data[str(i)]["ratios"]
        model.model.layers[i].self_attn.boundaries = label_data[str(i)]["boundary"]


def setup_lm_and_tokenizer(args, tokenizer_only=False):
    """Setup LM and tokenizer."""
    # Get model path
    model_path = MODEL_PATHS.get(Model[args.model_name], args.model_name)
    print(f"Loading model from: {model_path}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )

    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lm = None
    if not tokenizer_only:
        # Apply model modifications
        replace_gemma(args.method.lower())

        # Load model
        lm = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation
        )

        # Configure model layers
        _configure_attention_layers(args, lm)

        # Eval model
        lm.eval()

    return lm, tokenizer


def compute_ppl_with_full_response(
    model,
    tokenizer,
    prompt,
    response,
    max_len=None,
    return_loss=False,
    min_len=None
):
    """
    Compute perplexity with full response.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        prompt: The prompt to use.
        response: The response to use.
        max_len: The maximum length of the response.
        return_loss: Whether to return the loss.
        min_len: The minimum length of the response.

    Returns:
        A tuple of success and loss.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = tokenizer.model_max_length if max_len is None else max_len

    # 1) Tokenize the response alone to find its length
    resp_ids = tokenizer(
        response,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False
    )["input_ids"][0]
    resp_len = resp_ids.size(0)

    if resp_len >= max_len:
        raise ValueError(f"Response length ({resp_len}) ≥ context window ({max_len}); "
                         "you must truncate or split the response.")

    # 2) Tail‐truncate the prompt to leave exactly resp_len slots for the response
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False
    )["input_ids"][0]

    # how many prompt tokens we can keep
    keep_for_prompt = max_len - resp_len
    if prompt_ids.size(0) > keep_for_prompt:
        prompt_ids = prompt_ids[-keep_for_prompt:]

    # 3) Re‐assemble truncated prompt + full response
    input_ids = torch.cat([prompt_ids, resp_ids], dim=0).unsqueeze(0).to(device)

    if min_len is not None:
        if input_ids.size(1) < min_len:
            return False, None

    for layer in model.model.layers:
        layer.self_attn.instruction_len = None

    # 4) Mask out the prompt when computing loss
    if return_loss:
        labels = input_ids.clone()
        labels[0, :prompt_ids.size(0)] = -100

    # 5) Forward & exponentiate
    with torch.no_grad():
        if return_loss:
            labels = input_ids.clone()
            labels[0, :prompt_ids.size(0)] = -100
            outputs = model(
                input_ids,
                labels=labels,
                use_cache=False,
                output_hidden_states=False
            )
            loss = outputs.loss
            return True, loss.item()
        else:
            model(input_ids, use_cache=False, output_hidden_states=False)
            return True, None


def batched_forward(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_len: int | None = None,
    return_loss: bool = False,
):
    """
    • Tokenise prompts and responses separately.
    • Concatenate [prompt … response] for each pair.
    • If total length > max_len, drop tokens from the *start* of the prompt
      (response kept intact).
    • Right-pad every sequence to the longest sequence length in the batch.
    """
    assert len(prompts) == len(responses)

    device = next(model.parameters()).device
    max_len = tokenizer.model_max_length if max_len is None else max_len

    # Tokenise (no padding, no truncation yet)
    pr_tok = [tokenizer(prompt, return_tensors="pt",
                        truncation=False, add_special_tokens=False)["input_ids"][0]
              for prompt in prompts]
    resp_tok = [tokenizer(response, return_tensors="pt",
                          truncation=False, add_special_tokens=False)["input_ids"][0]
                for response in responses]

    input_ids_batch, label_ids_batch = [], []

    for p, r in zip(pr_tok, resp_tok):
        r_len = r.size(0)
        if r_len >= max_len:
            raise ValueError(f"Response length {r_len} ≥ context window {max_len}")

        concat = torch.cat([p, r], dim=0)
        excess = concat.size(0) - max_len
        if excess > 0:
            concat = concat[excess:]   # drop from front of prompt

        # Labels: mask prompt part
        lbl = concat.clone()
        prompt_kept = concat.size(0) - r_len
        lbl[:prompt_kept] = -100

        input_ids_batch.append(concat)
        label_ids_batch.append(lbl)

    # ---- dynamic padding to longest seq in the batch ----
    batch_len = max(seq.size(0) for seq in input_ids_batch)

    input_ids = torch.stack([
        pad(seq, (0, batch_len - seq.size(0)), value=tokenizer.pad_token_id)
        for seq in input_ids_batch
    ]).to(device)   # (B, batch_len)

    if return_loss:
        labels = torch.stack([
            pad(seq, (0, batch_len - seq.size(0)), value=-100)
            for seq in label_ids_batch
        ]).to(device)
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss
        return loss.item()
    else:
        with torch.no_grad():
            return model(input_ids)   # logits / model output


def reset_kv_cache(model, method: str) -> None:
    """Reset KV cache for streaming methods."""
    if method != SparseAttnMethod.full.name:
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.act_kv_seq_len = 0