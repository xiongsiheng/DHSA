import torch
import random
import numpy as np
import time
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple
import re
from transformers.cache_utils import Cache


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")

class FirstTokenTimer:
    def __init__(self, prompt_tokens: int):
        self.prompt_tokens = int(prompt_tokens)
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def put(self, value) -> None:
        if self.start_time is None or self.first_token_time is not None or value is None:
            return
        if value.ndim >= 2 and value.shape[-1] == self.prompt_tokens:
            return
        if value.numel() == 0:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.first_token_time = time.perf_counter()

    def end(self) -> None:
        pass

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.start_time is None or self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000.0


def infer_model_provider(model_name: str) -> str:
    """Infer the tokenizer/provider family from a Hugging Face model id or path."""
    normalized = model_name.lower()
    if "qwen2.5" in normalized:
        return "Qwen2.5"
    if "llama-3" in normalized:
        return "LLaMA3"
    raise ValueError(f"Unable to infer model provider from model name: {model_name}")


def preprocess_text(text: str) -> str:
    SINGLE_NEWLINE = re.compile(r'(?<!\n)\n(?!\n)')
    text = text.strip()  # Remove leading/trailing whitespace
    # normalise Windows line‑ends first (optional but nice)
    text = text.replace('\r\n', '\n')
    # 1) collapse single newlines into spaces
    text = SINGLE_NEWLINE.sub(' ', text)
    # 2) strip runaway whitespace if you like
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    return text


class SimpleQuantizedKVCache(Cache):
    """
    Minimal dependency-free KV cache.

    The cache stores each layer's K/V tensors as signed int8 plus one scale per
    tensor. For nbits < 8, values are quantized to the smaller numeric range but
    still stored in int8, so this saves the large bf16/fp16 cache allocation
    without adding bit-packing complexity.
    """

    def __init__(self, nbits: int = 4):
        super().__init__()
        if not 1 <= int(nbits) <= 8:
            raise ValueError("--nbits must be in [1, 8] for SimpleQuantizedKVCache")
        self.nbits = int(nbits)
        self.qmax = (1 << (self.nbits - 1)) - 1
        self.qmin = -self.qmax
        self.key_cache: List[Any] = []
        self.value_cache: List[Any] = []
        self.key_scales: List[Any] = []
        self.value_scales: List[Any] = []
        self.key_dtypes: List[Any] = []
        self.value_dtypes: List[Any] = []
        self.key_shapes: List[Any] = []
        self.value_shapes: List[Any] = []
        self._seen_tokens = 0

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
            self.key_scales.append([])
            self.value_scales.append([])
            self.key_dtypes.append(None)
            self.value_dtypes.append(None)
            self.key_shapes.append(None)
            self.value_shapes.append(None)

    @staticmethod
    def _pack_int4(q: torch.Tensor) -> torch.Tensor:
        flat = (q.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8).flatten()
        if flat.numel() % 2:
            flat = torch.cat((flat, flat.new_zeros(1)), dim=0)
        return ((flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)).contiguous()

    @staticmethod
    def _unpack_int4(packed: torch.Tensor, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        packed = packed.to(device=device)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        flat = torch.empty((packed.numel() * 2,), device=device, dtype=torch.uint8)
        flat[0::2] = low
        flat[1::2] = high
        total = math.prod(shape)
        return (flat[:total].to(torch.int16) - 8).to(torch.int8).view(shape)

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.dtype, Tuple[int, ...]]:
        dtype = x.dtype
        shape = tuple(x.shape)
        x_fp32 = x.detach().float()
        scale = x_fp32.abs().amax()
        if not torch.isfinite(scale) or scale.item() == 0.0:
            scale = torch.ones((), device=x.device, dtype=torch.float32)
        else:
            scale = scale / float(self.qmax)
        q = torch.clamp(torch.round(x_fp32 / scale), self.qmin, self.qmax).to(torch.int8)
        if self.nbits == 4:
            return self._pack_int4(q), scale.contiguous(), dtype, shape
        return q.contiguous(), scale.contiguous(), dtype, shape

    @staticmethod
    def _dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, shape: Tuple[int, ...], nbits: int) -> torch.Tensor:
        if nbits == 4:
            q = SimpleQuantizedKVCache._unpack_int4(q, shape, scale.device)
        return (q.float() * scale).to(dtype)

    def _get_layer_tensor(self, layer_idx: int, is_key: bool) -> Optional[torch.Tensor]:
        cache = self.key_cache if is_key else self.value_cache
        scales = self.key_scales if is_key else self.value_scales
        dtypes = self.key_dtypes if is_key else self.value_dtypes
        shapes = self.key_shapes if is_key else self.value_shapes
        if len(cache) <= layer_idx or cache[layer_idx] == []:
            return None
        return self._dequantize(cache[layer_idx], scales[layer_idx], dtypes[layer_idx], shapes[layer_idx], self.nbits)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        old_key = self._get_layer_tensor(layer_idx, is_key=True)
        old_value = self._get_layer_tensor(layer_idx, is_key=False)
        if old_key is not None:
            key_states = torch.cat((old_key, key_states), dim=-2)
            value_states = torch.cat((old_value, value_states), dim=-2)

        key_q, key_scale, key_dtype, key_shape = self._quantize(key_states)
        value_q, value_scale, value_dtype, value_shape = self._quantize(value_states)
        self.key_cache[layer_idx] = key_q
        self.value_cache[layer_idx] = value_q
        self.key_scales[layer_idx] = key_scale
        self.value_scales[layer_idx] = value_scale
        self.key_dtypes[layer_idx] = key_dtype
        self.value_dtypes[layer_idx] = value_dtype
        self.key_shapes[layer_idx] = key_shape
        self.value_shapes[layer_idx] = value_shape
        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] == []:
            return 0
        return int(self.key_shapes[layer_idx][-2])

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx] != []:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def __len__(self) -> int:
        return len(self.key_cache)