import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


LATEST_CHECKPOINT_REQUIRED_FIELDS = frozenset(
    {
        "predictor_config",
        "state_dict",
        "sample_prototypes",
    }
)
LATEST_CHECKPOINT_OPTIONAL_FIELDS = frozenset(
    {
        "density_config_overrides",
    }
)


def validate_latest_checkpoint(checkpoint: Mapping) -> None:
    """Reject incomplete checkpoints and pre-release compatibility metadata."""
    fields = set(checkpoint)
    missing = LATEST_CHECKPOINT_REQUIRED_FIELDS - fields
    if missing:
        raise ValueError(
            f"Latest checkpoint is missing fields: {sorted(missing)}"
        )
    unsupported = fields - (
        LATEST_CHECKPOINT_REQUIRED_FIELDS
        | LATEST_CHECKPOINT_OPTIONAL_FIELDS
    )
    if unsupported:
        raise ValueError(
            f"Latest checkpoint has unsupported fields: {sorted(unsupported)}"
        )


def resolve_density_config(
    predictor_config: Mapping,
    density_config_overrides: Mapping | None,
    density: float | None,
) -> dict:
    """Apply checkpoint overrides for the requested sparse density."""
    config = dict(predictor_config)
    if not density_config_overrides:
        return config
    if density is None:
        raise ValueError(
            "Checkpoint has density-specific configuration; density is required"
        )

    matches = [
        overrides
        for key, overrides in density_config_overrides.items()
        if math.isclose(
            float(key),
            float(density),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]
    if len(matches) != 1:
        supported = sorted(float(key) for key in density_config_overrides)
        raise ValueError(
            f"Density {density:g} has no unique checkpoint configuration; "
            f"supported densities are {supported}"
        )
    overrides = matches[0]
    if not isinstance(overrides, Mapping):
        raise ValueError(
            "Density-specific predictor configuration must be a mapping"
        )
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(
            f"Density-specific configuration has unknown keys: {sorted(unknown)}"
        )
    config.update(overrides)
    return config


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each feature vector by its root-mean-square magnitude."""
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class LowRankTopKScorer(nn.Module):
    """Compatibility model for the first predictor checkpoints."""

    def __init__(self, head_dim: int, rank: int = 32) -> None:
        """Initialize raw and low-rank query-key scoring projections."""
        super().__init__()
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.q_proj = nn.Linear(self.head_dim, self.rank, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.rank, bias=False)
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.learned_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        query_repr: torch.Tensor,
        key_repr: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        """Combine raw dot products with learned low-rank similarity scores."""
        if query_repr.shape[-1] != self.head_dim:
            raise ValueError("query_repr has the wrong head dimension")
        if key_repr.shape[-1] != self.head_dim:
            raise ValueError("key_repr has the wrong head dimension")

        point_query = query_repr.ndim + 1 == key_repr.ndim
        if point_query:
            score_equation = "...d,...kd->...k"
        elif query_repr.ndim == key_repr.ndim:
            score_equation = "...qd,...kd->...qk"
        else:
            raise ValueError("Unsupported query/key representation ranks")

        raw_scores = torch.einsum(
            score_equation,
            query_repr.float(),
            key_repr.float(),
        ) / math.sqrt(self.head_dim)
        projected_queries = self.q_proj(rms_normalize(query_repr.float()))
        projected_keys = self.k_proj(rms_normalize(key_repr.float()))
        learned_scores = torch.einsum(
            score_equation.replace("d", "r"),
            projected_queries,
            projected_keys,
        ) / math.sqrt(self.rank)
        return self.raw_scale * raw_scores + self.learned_scale * learned_scores


class _ConditionalLayerHeadMLP(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        local_feature_count: int,
        context_feature_count: int,
        hidden_size: int,
        depth: int,
        context_features: bool,
        num_samples: int,
        sample_conditioning: bool,
        sample_output_adapter: bool,
        sample_key_bias_buckets: int,
        sample_query_output_buckets: int,
        sample_budget_buckets: int,
    ) -> None:
        """Create one layer's head-specific MLP and optional sample adapters."""
        super().__init__()
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.context_features = bool(context_features)
        self.sample_conditioning = bool(sample_conditioning)
        self.sample_output_adapter = bool(sample_output_adapter)
        self.sample_key_bias_buckets = int(sample_key_bias_buckets)
        self.sample_query_output_buckets = int(sample_query_output_buckets)
        self.sample_budget_buckets = int(sample_budget_buckets)
        self.local_weight = nn.Parameter(
            torch.empty(
                self.num_heads,
                local_feature_count,
                self.hidden_size,
            )
        )
        if self.context_features:
            self.context_weight = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    context_feature_count,
                    self.hidden_size,
                )
            )
        else:
            self.register_parameter("context_weight", None)
        self.input_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size)
        )
        if self.sample_conditioning:
            self.sample_embedding = nn.Embedding(
                num_samples,
                self.num_heads * self.hidden_size,
            )
            nn.init.zeros_(self.sample_embedding.weight)
        else:
            self.sample_embedding = None
        if self.depth > 1:
            self.hidden_weight = nn.Parameter(
                torch.empty(
                    self.depth - 1,
                    self.num_heads,
                    self.hidden_size,
                    self.hidden_size,
                )
            )
            self.hidden_bias = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    self.num_heads,
                    self.hidden_size,
                )
            )
        else:
            self.register_parameter("hidden_weight", None)
            self.register_parameter("hidden_bias", None)
        self.output_weight = nn.Parameter(
            torch.zeros(self.num_heads, self.hidden_size)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.num_heads))
        if self.sample_output_adapter:
            self.sample_output_weight = nn.Embedding(
                num_samples,
                self.num_heads * self.hidden_size,
            )
            self.sample_output_bias = nn.Embedding(
                num_samples,
                self.num_heads,
            )
            nn.init.zeros_(self.sample_output_weight.weight)
            nn.init.zeros_(self.sample_output_bias.weight)
        else:
            self.sample_output_weight = None
            self.sample_output_bias = None
        if self.sample_key_bias_buckets > 0:
            self.sample_key_bias = nn.Embedding(
                num_samples,
                self.num_heads * self.sample_key_bias_buckets,
            )
            nn.init.zeros_(self.sample_key_bias.weight)
        else:
            self.sample_key_bias = None
        if self.sample_query_output_buckets > 0:
            self.sample_query_output_weight = nn.Embedding(
                num_samples,
                self.num_heads
                * self.sample_query_output_buckets
                * self.hidden_size,
            )
            nn.init.zeros_(self.sample_query_output_weight.weight)
        else:
            self.sample_query_output_weight = None
        if self.sample_budget_buckets > 0:
            self.sample_budget_bias = nn.Embedding(
                num_samples,
                self.num_heads * self.sample_budget_buckets,
            )
            nn.init.zeros_(self.sample_budget_bias.weight)
        else:
            self.sample_budget_bias = None
        self.reset_parameters(local_feature_count, context_feature_count)

    def reset_parameters(
        self,
        local_feature_count: int,
        context_feature_count: int,
    ) -> None:
        """Initialize MLP weights with fan-in-scaled normal distributions."""
        nn.init.normal_(
            self.local_weight,
            std=1.0 / math.sqrt(local_feature_count),
        )
        if self.context_weight is not None:
            nn.init.normal_(
                self.context_weight,
                std=1.0 / math.sqrt(context_feature_count),
            )
        if self.hidden_weight is not None:
            nn.init.normal_(
                self.hidden_weight,
                std=1.0 / math.sqrt(self.hidden_size),
            )


class ConditionalTopKMLP(nn.Module):
    """Query-conditioned nonlinear scorer used by memorization checkpoints."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        depth: int,
        residual_bound: float,
        context_features: bool,
        pair_count: int | None = None,
        head_dim: int = 128,
        query_segments: int = 4,
        key_segments: int = 2,
        mlp_key_chunk_size: int = 256,
        num_samples: int = 0,
        sample_conditioning: bool = False,
        sample_output_adapter: bool = False,
        sample_key_bias_buckets: int = 0,
        sample_query_output_buckets: int = 0,
        sample_budget_buckets: int = 0,
        dynamic_density_min_ratio: float = 0.5,
        dynamic_density_max_ratio: float = 2.0,
        key_bias_only: bool = False,
    ) -> None:
        """Configure the current conditional predictor architecture."""
        super().__init__()
        if min(
            num_layers,
            num_heads,
            hidden_size,
            depth,
            head_dim,
            query_segments,
            key_segments,
            mlp_key_chunk_size,
        ) <= 0:
            raise ValueError("All predictor dimensions must be positive")
        if residual_bound <= 0:
            raise ValueError("residual_bound must be positive")
        expected_pair_count = int(query_segments) * int(key_segments)
        if pair_count is not None and int(pair_count) != expected_pair_count:
            raise ValueError(
                f"pair_count={pair_count} does not match "
                f"{query_segments}x{key_segments} segments"
            )
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.residual_bound = float(residual_bound)
        self.context_features = bool(context_features)
        self.head_dim = int(head_dim)
        self.query_segments = int(query_segments)
        self.key_segments = int(key_segments)
        self.pair_count = expected_pair_count
        self.mlp_key_chunk_size = int(mlp_key_chunk_size)
        self.num_samples = int(num_samples)
        self.sample_conditioning = bool(sample_conditioning)
        self.sample_output_adapter = bool(sample_output_adapter)
        self.sample_key_bias_buckets = int(sample_key_bias_buckets)
        self.sample_query_output_buckets = int(sample_query_output_buckets)
        self.sample_budget_buckets = int(sample_budget_buckets)
        self.dynamic_density_min_ratio = float(dynamic_density_min_ratio)
        self.dynamic_density_max_ratio = float(dynamic_density_max_ratio)
        self.key_bias_only = bool(key_bias_only)
        if self.sample_key_bias_buckets < 0:
            raise ValueError("sample_key_bias_buckets must be non-negative")
        if self.sample_query_output_buckets < 0:
            raise ValueError("sample_query_output_buckets must be non-negative")
        if self.sample_budget_buckets < 0:
            raise ValueError("sample_budget_buckets must be non-negative")
        if not 0.0 <= self.dynamic_density_min_ratio <= 1.0:
            raise ValueError("dynamic_density_min_ratio must be in [0, 1]")
        if self.dynamic_density_max_ratio < 1.0:
            raise ValueError("dynamic_density_max_ratio must be at least 1")
        if self.key_bias_only and self.sample_key_bias_buckets <= 0:
            raise ValueError(
                "key_bias_only requires positive sample_key_bias_buckets"
            )
        if (
            self.sample_conditioning
            or self.sample_output_adapter
            or self.sample_key_bias_buckets > 0
            or self.sample_query_output_buckets > 0
            or self.sample_budget_buckets > 0
        ) and self.num_samples <= 0:
            raise ValueError(
                "sample-specific parameters require a positive num_samples"
            )
        self.local_feature_count = self.pair_count + 7
        self.context_feature_count = self.pair_count * 4
        self.layers = nn.ModuleList(
            _ConditionalLayerHeadMLP(
                num_heads=self.num_heads,
                local_feature_count=self.local_feature_count,
                context_feature_count=self.context_feature_count,
                hidden_size=self.hidden_size,
                depth=self.depth,
                context_features=self.context_features,
                num_samples=self.num_samples,
                sample_conditioning=self.sample_conditioning,
                sample_output_adapter=self.sample_output_adapter,
                sample_key_bias_buckets=self.sample_key_bias_buckets,
                sample_query_output_buckets=self.sample_query_output_buckets,
                sample_budget_buckets=self.sample_budget_buckets,
            )
            for _ in range(self.num_layers)
        )
        self.register_buffer(
            "_sample_prototypes",
            None,
            persistent=False,
        )
        self.prototype_query_start_fraction = 0.0625
        self._cached_sample_indices = None
        self._cached_sample_shape = None

    def export_config(self) -> dict:
        """Return all constructor parameters needed to rebuild this predictor."""
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "residual_bound": self.residual_bound,
            "context_features": self.context_features,
            "pair_count": self.pair_count,
            "head_dim": self.head_dim,
            "query_segments": self.query_segments,
            "key_segments": self.key_segments,
            "mlp_key_chunk_size": self.mlp_key_chunk_size,
            "num_samples": self.num_samples,
            "sample_conditioning": self.sample_conditioning,
            "sample_output_adapter": self.sample_output_adapter,
            "sample_key_bias_buckets": self.sample_key_bias_buckets,
            "sample_query_output_buckets": self.sample_query_output_buckets,
            "sample_budget_buckets": self.sample_budget_buckets,
            "dynamic_density_min_ratio": self.dynamic_density_min_ratio,
            "dynamic_density_max_ratio": self.dynamic_density_max_ratio,
            "key_bias_only": self.key_bias_only,
        }

    def set_sample_prototypes(
        self,
        prototypes: torch.Tensor,
        *,
        query_start_fraction: float = 0.0625,
    ) -> None:
        """Attach sample signatures used for nearest-prototype lookup."""
        if prototypes.ndim != 5:
            raise ValueError(
                "sample prototypes must have shape [L,N,H,Q,F]"
            )
        expected_prefix = (
            self.num_layers,
            self.num_samples,
            self.num_heads,
        )
        if prototypes.shape[:3] != expected_prefix:
            raise ValueError(
                f"sample prototype prefix {prototypes.shape[:3]} does not "
                f"match {expected_prefix}"
            )
        if prototypes.shape[-1] != self.context_feature_count:
            raise ValueError("sample prototypes have the wrong feature count")
        if not 0.0 <= query_start_fraction < 1.0:
            raise ValueError("query_start_fraction must be in [0, 1)")
        self._sample_prototypes = prototypes
        self.prototype_query_start_fraction = float(query_start_fraction)
        self.clear_sample_match_cache()

    def clear_sample_match_cache(self) -> None:
        """Clear cached nearest-prototype IDs after inputs or prototypes change."""
        self._cached_sample_indices = None
        self._cached_sample_shape = None

    def _prototype_query_ids(
        self,
        num_query_blocks: int,
        prototype_query_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Choose prompt-relative query rows for prototype comparison."""
        first_query_block = max(
            0,
            min(
                num_query_blocks - 1,
                round(
                    num_query_blocks
                    * self.prototype_query_start_fraction
                ),
            ),
        )
        # Short prompts use nearest-neighbor repeats to preserve checkpoint shape.
        query_ids = torch.linspace(
            first_query_block,
            num_query_blocks - 1,
            steps=prototype_query_blocks,
            device=device,
        ).round().long()
        return query_ids

    def _match_sample_prototypes(
        self,
        context: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Return the nearest stored prototype index for each batch item."""
        if self._sample_prototypes is None:
            raise ValueError(
                "sample-conditioned checkpoint has no runtime prototypes"
            )
        prototypes = self._sample_prototypes[layer].to(
            device=context.device,
            dtype=torch.float32,
        )
        query_ids = self._prototype_query_ids(
            context.shape[2],
            prototypes.shape[-2],
            context.device,
        )
        signature = context[:, :, query_ids, :].float()
        squared_distance = (
            signature.unsqueeze(1) - prototypes.unsqueeze(0)
        ).square().mean(dim=(-1, -2, -3))
        return squared_distance.argmin(dim=-1)

    @staticmethod
    def _layer_index(layer_idx: int | torch.Tensor, num_layers: int) -> int:
        """Convert and validate a scalar predictor layer index."""
        if isinstance(layer_idx, torch.Tensor):
            if layer_idx.numel() != 1:
                raise ValueError("A forward call must contain one layer")
            layer = int(layer_idx.item())
        else:
            layer = int(layer_idx)
        if not 0 <= layer < num_layers:
            raise ValueError(
                f"layer {layer} is outside predictor range [0, {num_layers})"
            )
        return layer

    @staticmethod
    def _expand_valid_counts(
        valid_counts: torch.Tensor,
        batch_size: int,
        num_query_blocks: int,
    ) -> torch.Tensor:
        """Broadcast per-query valid-key counts across the batch when needed."""
        if valid_counts.ndim == 1:
            if valid_counts.numel() != num_query_blocks:
                raise ValueError(
                    "valid_counts must contain one value per query block"
                )
            return valid_counts.view(1, -1).expand(batch_size, -1)
        if valid_counts.shape != (batch_size, num_query_blocks):
            raise ValueError(
                "valid_counts must have shape [Q] or [B,Q]"
            )
        return valid_counts

    @staticmethod
    def _position_features(
        valid_counts: torch.Tensor,
        num_key_blocks: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode normalized age, log age, and recency for each key block."""
        key_positions = torch.arange(
            num_key_blocks,
            device=valid_counts.device,
            dtype=dtype,
        )
        valid = valid_counts.to(dtype=dtype).unsqueeze(-1)
        age = (valid - 1.0 - key_positions).clamp_min(0.0)
        normalized_age = age / (valid - 1.0).clamp_min(1.0)
        log_age = torch.log1p(age) / torch.log1p(valid).clamp_min(1.0)
        recentness = torch.rsqrt(age + 1.0)
        return torch.stack(
            (normalized_age, log_age, recentness),
            dim=-1,
        )

    @staticmethod
    def _masked_key_stats(
        values: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute valid-key mean and standard deviation for feature tensors."""
        num_key_blocks = values.shape[-2]
        valid = (
            torch.arange(num_key_blocks, device=values.device).view(1, 1, -1)
            < valid_counts.unsqueeze(-1)
        )
        mask = valid.unsqueeze(1).unsqueeze(-1).to(dtype=values.dtype)
        divisor = valid_counts.clamp_min(1).to(dtype=values.dtype)
        divisor = divisor.unsqueeze(1).unsqueeze(-1)
        mean = (values * mask).sum(dim=-2) / divisor
        variance = (
            (values - mean.unsqueeze(-2)).square() * mask
        ).sum(dim=-2) / divisor
        return mean, variance.clamp_min(0.0).sqrt()

    def _context(
        self,
        pair_scores: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> torch.Tensor | None:
        """Assemble global pair-score statistics for conditional prediction."""
        if not self.context_features:
            return None
        pair_mean, pair_std = self._masked_key_stats(
            pair_scores,
            valid_counts,
        )
        sink = pair_scores[:, :, :, 0, :]
        recent_indices = (
            valid_counts.clamp(min=1, max=pair_scores.shape[-2]) - 1
        )
        recent_indices = recent_indices[:, None, :, None, None].expand(
            pair_scores.shape[0],
            pair_scores.shape[1],
            pair_scores.shape[2],
            1,
            pair_scores.shape[-1],
        )
        recent = pair_scores.gather(-2, recent_indices).squeeze(-2)
        return torch.cat((pair_mean, pair_std, sink, recent), dim=-1)

    def _apply_mlp(
        self,
        local: torch.Tensor,
        context: torch.Tensor | None,
        layer_mlp: _ConditionalLayerHeadMLP,
        sample_embedding: torch.Tensor | None,
        sample_output_weight: torch.Tensor | None,
        sample_output_bias: torch.Tensor | None,
        sample_query_output_weight: torch.Tensor | None,
        query_buckets: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the selected layer/head MLP and sample-specific adapters."""
        hidden = torch.einsum(
            "bhqkf,hfd->bhqkd",
            local,
            layer_mlp.local_weight,
        )
        hidden = hidden + layer_mlp.input_bias.view(
            1,
            self.num_heads,
            1,
            1,
            self.hidden_size,
        )
        if sample_embedding is not None:
            hidden = hidden + sample_embedding[:, :, None, None, :]
        if context is not None:
            hidden = hidden + torch.einsum(
                "bhqf,hfd->bhqd",
                context,
                layer_mlp.context_weight,
            ).unsqueeze(-2)
        hidden = F.silu(hidden)
        for depth_index in range(self.depth - 1):
            hidden = F.silu(
                torch.einsum(
                    "bhqkd,hde->bhqke",
                    hidden,
                    layer_mlp.hidden_weight[depth_index],
                )
                + layer_mlp.hidden_bias[depth_index].view(
                    1,
                    self.num_heads,
                    1,
                    1,
                    self.hidden_size,
                )
            )
        raw_residual = torch.einsum(
            "bhqkd,hd->bhqk",
            hidden,
            layer_mlp.output_weight,
        )
        raw_residual = raw_residual + layer_mlp.output_bias.view(1, -1, 1, 1)
        if sample_output_weight is not None:
            raw_residual = raw_residual + torch.einsum(
                "bhqkd,bhd->bhqk",
                hidden,
                sample_output_weight,
            )
            raw_residual = raw_residual + sample_output_bias[:, :, None, None]
        if sample_query_output_weight is not None:
            query_indices = query_buckets[:, None, :, None].expand(
                -1,
                self.num_heads,
                -1,
                self.hidden_size,
            )
            query_weights = sample_query_output_weight.gather(
                2,
                query_indices,
            )
            raw_residual = raw_residual + torch.einsum(
                "bhqkd,bhqd->bhqk",
                hidden,
                query_weights,
            )
        return raw_residual

    def _resolve_sample_indices(
        self,
        pair_scores: torch.Tensor,
        expanded_valid_counts: torch.Tensor,
        layer: int,
        sample_idx: int | torch.Tensor | None,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Resolve explicit or nearest-prototype sample IDs and cache matches."""
        if sample_idx is None:
            sample_shape = (
                pair_scores.shape[0],
                pair_scores.shape[2],
                pair_scores.shape[3],
                pair_scores.device,
            )
            needs_match = (
                layer == 0
                or self._cached_sample_indices is None
                or self._cached_sample_shape != sample_shape
            )
            if needs_match:
                if context is None:
                    if not self.context_features:
                        raise ValueError(
                            "runtime prototype matching requires context features"
                        )
                    context = self._context(
                        pair_scores,
                        expanded_valid_counts,
                    )
                self._cached_sample_indices = (
                    self._match_sample_prototypes(context, layer)
                )
                self._cached_sample_shape = sample_shape
            sample_indices = self._cached_sample_indices
        else:
            sample_indices = torch.as_tensor(
                sample_idx,
                device=pair_scores.device,
                dtype=torch.long,
            ).reshape(-1)
        if sample_indices.numel() == 1:
            sample_indices = sample_indices.expand(pair_scores.shape[0])
        if sample_indices.numel() != pair_scores.shape[0]:
            raise ValueError("sample_idx must contain one ID per batch item")
        if bool(
            ((sample_indices < 0) | (sample_indices >= self.num_samples)).any()
        ):
            raise ValueError("sample_idx is outside the checkpoint range")
        return sample_indices, context

    def _sample_budget_logits(
        self,
        layer_mlp: _ConditionalLayerHeadMLP,
        sample_indices: torch.Tensor | None,
        valid_counts: torch.Tensor,
        num_key_blocks: int,
    ) -> torch.Tensor | None:
        """Gather per-query dynamic budget logits for matched samples."""
        if self.sample_budget_buckets <= 0:
            return None
        budget_table = layer_mlp.sample_budget_bias(sample_indices).view(
            valid_counts.shape[0],
            self.num_heads,
            self.sample_budget_buckets,
        )
        query_buckets = (
            (
                (valid_counts.float() - 1.0)
                / max(num_key_blocks - 1, 1)
                * self.sample_budget_buckets
            )
            .floor()
            .long()
            .clamp(min=0, max=self.sample_budget_buckets - 1)
        )
        return budget_table.gather(
            2,
            query_buckets[:, None, :].expand(
                -1,
                self.num_heads,
                -1,
            ),
        )

    def score_pair_features(
        self,
        pair_scores: torch.Tensor,
        *,
        layer_idx: int | torch.Tensor,
        valid_counts: torch.Tensor,
        sample_idx: int | torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score precomputed query/key segment pairs for one model layer."""
        squeeze_batch = pair_scores.ndim == 4
        if squeeze_batch:
            pair_scores = pair_scores.unsqueeze(0)
        if pair_scores.ndim != 5:
            raise ValueError("pair_scores must have shape [H,Q,K,P] or [B,H,Q,K,P]")
        if pair_scores.shape[1] != self.num_heads:
            raise ValueError("pair_scores has the wrong head count")
        if pair_scores.shape[-1] != self.pair_count:
            raise ValueError("pair_scores has the wrong segment-pair count")
        layer = self._layer_index(layer_idx, self.num_layers)
        layer_mlp = self.layers[layer]
        compute_dtype = layer_mlp.local_weight.dtype
        pair_scores = pair_scores.to(dtype=compute_dtype)
        expanded_valid_counts = self._expand_valid_counts(
            valid_counts.to(device=pair_scores.device),
            pair_scores.shape[0],
            pair_scores.shape[2],
        )
        base_scores = torch.logsumexp(pair_scores, dim=-1) - math.log(
            self.pair_count
        )
        context = (
            None
            if self.key_bias_only
            else self._context(pair_scores, expanded_valid_counts)
        )
        if (
            self.sample_conditioning
            or self.sample_output_adapter
            or self.sample_key_bias_buckets > 0
            or self.sample_query_output_buckets > 0
            or self.sample_budget_buckets > 0
        ):
            sample_indices, context = self._resolve_sample_indices(
                pair_scores,
                expanded_valid_counts,
                layer,
                sample_idx,
                context,
            )
        else:
            sample_indices = None
        budget_logits = self._sample_budget_logits(
            layer_mlp,
            sample_indices,
            expanded_valid_counts,
            pair_scores.shape[-2],
        )
        if self.key_bias_only:
            sample_key_bias = layer_mlp.sample_key_bias(
                sample_indices
            ).view(
                pair_scores.shape[0],
                self.num_heads,
                self.sample_key_bias_buckets,
            )
            key_buckets = (
                torch.arange(
                    pair_scores.shape[-2],
                    device=pair_scores.device,
                )
                % self.sample_key_bias_buckets
            )
            residual = self.residual_bound * torch.tanh(
                sample_key_bias[:, :, None, key_buckets]
            )
            scores = base_scores + residual
            if squeeze_batch:
                scores = scores.squeeze(0)
                base_scores = base_scores.squeeze(0)
                if budget_logits is not None:
                    budget_logits = budget_logits.squeeze(0)
            if return_aux:
                auxiliary = {
                    "base_scores": base_scores,
                    "residual": scores - base_scores,
                    "sample_indices": sample_indices,
                }
                if budget_logits is not None:
                    auxiliary["budget_logits"] = budget_logits
                return scores, auxiliary
            return scores
        if self.sample_conditioning:
            sample_embedding = layer_mlp.sample_embedding(
                sample_indices
            ).view(
                pair_scores.shape[0],
                self.num_heads,
                self.hidden_size,
            )
        else:
            sample_embedding = None
        if self.sample_output_adapter:
            sample_output_weight = layer_mlp.sample_output_weight(
                sample_indices
            ).view(
                pair_scores.shape[0],
                self.num_heads,
                self.hidden_size,
            )
            sample_output_bias = layer_mlp.sample_output_bias(sample_indices)
        else:
            sample_output_weight = None
            sample_output_bias = None
        if self.sample_key_bias_buckets > 0:
            sample_key_bias = layer_mlp.sample_key_bias(
                sample_indices
            ).view(
                pair_scores.shape[0],
                self.num_heads,
                self.sample_key_bias_buckets,
            )
        else:
            sample_key_bias = None
        if self.sample_query_output_buckets > 0:
            sample_query_output_weight = (
                layer_mlp.sample_query_output_weight(sample_indices).view(
                    pair_scores.shape[0],
                    self.num_heads,
                    self.sample_query_output_buckets,
                    self.hidden_size,
                )
            )
            query_buckets = (
                (
                    (expanded_valid_counts.float() - 1.0)
                    / max(pair_scores.shape[-2] - 1, 1)
                    * self.sample_query_output_buckets
                )
                .floor()
                .long()
                .clamp(max=self.sample_query_output_buckets - 1)
            )
        else:
            sample_query_output_weight = None
            query_buckets = None
        positions = self._position_features(
            expanded_valid_counts,
            pair_scores.shape[-2],
            pair_scores.dtype,
        )
        score_chunks = []
        for key_start in range(0, pair_scores.shape[-2], self.mlp_key_chunk_size):
            key_end = min(
                pair_scores.shape[-2],
                key_start + self.mlp_key_chunk_size,
            )
            pair_chunk = pair_scores[..., key_start:key_end, :]
            base_chunk = base_scores[..., key_start:key_end]
            centered_pairs = torch.tanh(
                pair_chunk - base_chunk.unsqueeze(-1)
            )
            pair_stats = torch.stack(
                (
                    pair_chunk.std(dim=-1, unbiased=False),
                    pair_chunk.amax(dim=-1) - base_chunk,
                    base_chunk - pair_chunk.amin(dim=-1),
                ),
                dim=-1,
            ).tanh()
            position_chunk = positions[:, None, :, key_start:key_end, :]
            position_chunk = position_chunk.expand(
                -1,
                self.num_heads,
                -1,
                -1,
                -1,
            )
            local = torch.cat(
                (
                    centered_pairs,
                    base_chunk.unsqueeze(-1).tanh(),
                    pair_stats,
                    position_chunk,
                ),
                dim=-1,
            )
            raw_residual = self._apply_mlp(
                local,
                context,
                layer_mlp,
                sample_embedding,
                sample_output_weight,
                sample_output_bias,
                sample_query_output_weight,
                query_buckets,
            )
            if sample_key_bias is not None:
                key_buckets = (
                    torch.arange(
                        key_start,
                        key_end,
                        device=pair_scores.device,
                    )
                    % self.sample_key_bias_buckets
                )
                raw_residual = raw_residual + sample_key_bias[
                    :, :, None, key_buckets
                ]
            residual = self.residual_bound * torch.tanh(raw_residual)
            score_chunks.append(base_chunk + residual)
        scores = torch.cat(score_chunks, dim=-1)
        if squeeze_batch:
            scores = scores.squeeze(0)
            base_scores = base_scores.squeeze(0)
            if budget_logits is not None:
                budget_logits = budget_logits.squeeze(0)
        if return_aux:
            auxiliary = {
                "base_scores": base_scores,
                "residual": scores - base_scores,
            }
            if sample_indices is not None:
                auxiliary["sample_indices"] = sample_indices
            if budget_logits is not None:
                auxiliary["budget_logits"] = budget_logits
            return scores, auxiliary
        return scores

    def forward(
        self,
        query_repr: torch.Tensor,
        key_repr: torch.Tensor,
        *,
        layer_idx: int | torch.Tensor,
        valid_counts: torch.Tensor,
        sample_idx: int | torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build segment-pair features and return learned key-block scores."""
        if query_repr.ndim != 5 or key_repr.ndim != 5:
            raise ValueError(
                "ConditionalTopKMLP expects Q=[B,H,Q,Sq,D] and "
                "K=[B,H,K,Sk,D]"
            )
        if query_repr.shape[:2] != key_repr.shape[:2]:
            raise ValueError("query/key batch and head dimensions must match")
        if query_repr.shape[1] != self.num_heads:
            raise ValueError("query_repr has the wrong head count")
        if query_repr.shape[-2:] != (self.query_segments, self.head_dim):
            raise ValueError("query_repr has the wrong segment/head dimensions")
        if key_repr.shape[-2:] != (self.key_segments, self.head_dim):
            raise ValueError("key_repr has the wrong segment/head dimensions")
        compute_dtype = self.layers[0].local_weight.dtype
        query_compute = query_repr.to(dtype=compute_dtype)
        key_compute = key_repr.to(dtype=compute_dtype)
        pair_scores = []
        divisor = math.sqrt(self.head_dim)
        for query_segment in range(self.query_segments):
            for key_segment in range(self.key_segments):
                pair_scores.append(
                    torch.einsum(
                        "bhqd,bhkd->bhqk",
                        query_compute[..., query_segment, :],
                        key_compute[..., key_segment, :],
                    )
                    / divisor
                )
        return self.score_pair_features(
            torch.stack(pair_scores, dim=-1),
            layer_idx=layer_idx,
            valid_counts=valid_counts,
            sample_idx=sample_idx,
            return_aux=return_aux,
        )


class BlockTopKPredictor(nn.Module):
    """Small layer/head-conditioned scorer for block-level TopK routing."""

    def __init__(
        self,
        head_dim: int,
        rank: int,
        num_layers: int,
        num_heads: int,
        use_raw_scores: bool = True,
        query_segments: int = 1,
        key_segments: int = 1,
        full_segment_scores: bool = False,
        bounded_residual: bool = False,
        residual_bound: float = 0.25,
        residual_rank: int = 0,
        streaming_residual: bool = False,
        residual_use_stats: bool = True,
        fused_residual: bool = False,
    ) -> None:
        """Initialize the earlier block scorer and its optional residual modes."""
        super().__init__()
        if min(
            head_dim,
            rank,
            num_layers,
            num_heads,
            query_segments,
            key_segments,
        ) <= 0:
            raise ValueError("All predictor dimensions must be positive")
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.use_raw_scores = bool(use_raw_scores)
        self.query_segments = int(query_segments)
        self.key_segments = int(key_segments)
        self.full_segment_scores = bool(full_segment_scores)
        self.bounded_residual = bool(bounded_residual)
        self.residual_bound = float(residual_bound)
        self.residual_rank = int(residual_rank)
        self.streaming_residual = bool(streaming_residual)
        self.residual_use_stats = bool(residual_use_stats)
        self.fused_residual = bool(fused_residual)
        if self.bounded_residual and not self.full_segment_scores:
            raise ValueError("bounded_residual requires full_segment_scores")
        if self.bounded_residual and self.use_raw_scores:
            raise ValueError("bounded_residual requires use_raw_scores=False")
        if self.residual_bound <= 0.0:
            raise ValueError("residual_bound must be positive")
        if self.residual_rank < 0:
            raise ValueError("residual_rank must be non-negative")
        if self.residual_rank and not self.bounded_residual:
            raise ValueError("residual_rank requires bounded_residual")
        if self.streaming_residual and not self.bounded_residual:
            raise ValueError("streaming_residual requires bounded_residual")
        if self.streaming_residual and self.residual_rank:
            raise ValueError("streaming_residual does not support residual_rank")
        if self.fused_residual and not self.bounded_residual:
            raise ValueError("fused_residual requires bounded_residual")
        if self.fused_residual and self.streaming_residual:
            raise ValueError("fused_residual and streaming_residual are exclusive")
        if self.fused_residual and self.residual_rank:
            raise ValueError("fused_residual does not support residual_rank")
        if (
            not self.streaming_residual
            and not self.fused_residual
            and not self.residual_use_stats
        ):
            raise ValueError(
                "residual_use_stats=False requires a streaming residual mode"
            )

        if self.full_segment_scores:
            self.q_proj = None
            self.k_proj = None
        else:
            self.q_proj = nn.Linear(self.head_dim, self.rank, bias=False)
            self.k_proj = nn.Linear(self.head_dim, self.rank, bias=False)
            with torch.no_grad():
                projection = torch.empty_like(self.q_proj.weight).bernoulli_(0.5)
                projection.mul_(2.0).sub_(1.0).div_(math.sqrt(self.rank))
                self.q_proj.weight.copy_(projection)
                self.k_proj.weight.copy_(projection)
        if self.bounded_residual:
            self.register_parameter("log_raw_scale", None)
            self.register_parameter("log_learned_scale", None)
            self.register_parameter("position_weight", None)
            self.register_parameter("log_temperature", None)
            self.register_parameter("segment_bias", None)
            self.residual_pair_weight = nn.Parameter(
                torch.zeros(
                    self.num_layers,
                    self.num_heads,
                    self.query_segments,
                    self.key_segments,
                )
            )
            if self.residual_use_stats:
                self.residual_stat_weight = nn.Parameter(
                    torch.zeros(self.num_layers, self.num_heads, 3)
                )
            else:
                self.register_parameter("residual_stat_weight", None)
            self.residual_position_weight = nn.Parameter(
                torch.zeros(self.num_layers, self.num_heads, 3)
            )
            if self.residual_rank:
                projection = torch.empty(
                    self.num_layers,
                    self.head_dim,
                    self.residual_rank,
                ).bernoulli_(0.5)
                projection.mul_(2.0).sub_(1.0).div_(
                    math.sqrt(self.residual_rank)
                )
                self.residual_q_proj = nn.Parameter(projection.clone())
                self.residual_k_proj = nn.Parameter(projection.clone())
                self.residual_low_rank_weight = nn.Parameter(
                    torch.zeros(
                        self.num_layers,
                        self.num_heads,
                        self.query_segments,
                        self.key_segments,
                    )
                )
            else:
                self.register_parameter("residual_q_proj", None)
                self.register_parameter("residual_k_proj", None)
                self.register_parameter("residual_low_rank_weight", None)
        else:
            self.register_parameter("residual_pair_weight", None)
            self.register_parameter("residual_stat_weight", None)
            self.register_parameter("residual_position_weight", None)
            self.register_parameter("residual_q_proj", None)
            self.register_parameter("residual_k_proj", None)
            self.register_parameter("residual_low_rank_weight", None)
            if self.use_raw_scores:
                self.log_raw_scale = nn.Parameter(
                    torch.zeros(self.num_layers, self.num_heads)
                )
                learned_scale_init = math.log(0.1)
            else:
                self.register_parameter("log_raw_scale", None)
                learned_scale_init = 0.0
            self.log_learned_scale = nn.Parameter(
                torch.full(
                    (self.num_layers, self.num_heads),
                    learned_scale_init,
                )
            )
            self.position_weight = nn.Parameter(
                torch.zeros(self.num_layers, self.num_heads, 3)
            )
            self.log_temperature = nn.Parameter(
                torch.zeros(self.num_layers, self.num_heads)
            )
            self.segment_bias = nn.Parameter(
                torch.zeros(
                    self.num_layers,
                    self.num_heads,
                    self.query_segments,
                    self.key_segments,
                )
            )

    def export_config(self) -> dict:
        """Return constructor parameters for rebuilding this block predictor."""
        return {
            "head_dim": self.head_dim,
            "rank": self.rank,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "use_raw_scores": self.use_raw_scores,
            "query_segments": self.query_segments,
            "key_segments": self.key_segments,
            "full_segment_scores": self.full_segment_scores,
            "bounded_residual": self.bounded_residual,
            "residual_bound": self.residual_bound,
            "residual_rank": self.residual_rank,
            "streaming_residual": self.streaming_residual,
            "residual_use_stats": self.residual_use_stats,
            "fused_residual": self.fused_residual,
        }

    @staticmethod
    def _position_features(
        valid_counts: torch.Tensor,
        num_key_blocks: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode normalized age, log age, and recency for each key block."""
        key_positions = torch.arange(
            num_key_blocks,
            device=valid_counts.device,
            dtype=dtype,
        )
        valid = valid_counts.to(dtype=dtype).unsqueeze(-1)
        age = (valid - 1.0 - key_positions).clamp_min(0.0)
        normalized_age = age / (valid - 1.0).clamp_min(1.0)
        log_age = torch.log1p(age) / torch.log1p(valid).clamp_min(1.0)
        recentness = torch.rsqrt(age + 1.0)
        return torch.stack((normalized_age, log_age, recentness), dim=-1)

    def forward(
        self,
        query_repr: torch.Tensor,
        key_repr: torch.Tensor,
        *,
        layer_idx: int | torch.Tensor,
        valid_counts: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score key blocks using projected similarity and positional residuals."""
        if query_repr.ndim == 4:
            query_repr = query_repr.unsqueeze(-2)
        if key_repr.ndim == 4:
            key_repr = key_repr.unsqueeze(-2)
        if query_repr.ndim != 5 or key_repr.ndim != 5:
            raise ValueError(
                "BlockTopKPredictor expects Q=[B,H,Q,Sq,D] and "
                "K=[B,H,K,Sk,D]"
            )
        if query_repr.shape[:2] != key_repr.shape[:2]:
            raise ValueError("query/key batch and head dimensions must match")
        if query_repr.shape[-1] != self.head_dim:
            raise ValueError("query_repr has the wrong head dimension")
        if key_repr.shape[-1] != self.head_dim:
            raise ValueError("key_repr has the wrong head dimension")
        if query_repr.shape[1] != self.num_heads:
            raise ValueError(
                f"checkpoint expects {self.num_heads} heads, "
                f"received {query_repr.shape[1]}"
            )
        if query_repr.shape[-2] != self.query_segments:
            raise ValueError(
                f"checkpoint expects {self.query_segments} query segments, "
                f"received {query_repr.shape[-2]}"
            )
        if key_repr.shape[-2] != self.key_segments:
            raise ValueError(
                f"checkpoint expects {self.key_segments} key segments, "
                f"received {key_repr.shape[-2]}"
            )
        if valid_counts.ndim != 1 or valid_counts.numel() != query_repr.shape[2]:
            raise ValueError("valid_counts must contain one value per query block")

        if isinstance(layer_idx, torch.Tensor):
            if layer_idx.numel() != 1:
                raise ValueError("A forward call must contain one layer")
            layer = int(layer_idx.item())
        else:
            layer = int(layer_idx)
        if not 0 <= layer < self.num_layers:
            raise ValueError(
                f"layer {layer} is outside predictor range [0, {self.num_layers})"
            )

        if self.bounded_residual:
            compute_dtype = self.residual_pair_weight.dtype
        elif self.full_segment_scores:
            compute_dtype = self.segment_bias.dtype
        else:
            compute_dtype = self.q_proj.weight.dtype
        query_compute = query_repr.to(dtype=compute_dtype)
        key_compute = key_repr.to(dtype=compute_dtype)
        if self.full_segment_scores:
            projected_queries = query_compute
            projected_keys = key_compute
            score_divisor = math.sqrt(self.head_dim)
        else:
            projected_queries = self.q_proj(query_compute)
            projected_keys = self.k_proj(key_compute)
            score_divisor = math.sqrt(self.head_dim)

        if self.bounded_residual and self.fused_residual:
            base_aggregate = None
            adjusted_aggregate = None
            pair_biases = (
                0.5
                * self.residual_bound
                * torch.tanh(self.residual_pair_weight[layer])
            )
            for query_segment in range(self.query_segments):
                for key_segment in range(self.key_segments):
                    pair_score = (
                        torch.einsum(
                            "bhqr,bhkr->bhqk",
                            projected_queries[..., query_segment, :],
                            projected_keys[..., key_segment, :],
                        )
                        / score_divisor
                    )
                    adjusted_pair = pair_score + pair_biases[
                        :,
                        query_segment,
                        key_segment,
                    ].to(dtype=compute_dtype).view(1, -1, 1, 1)
                    adjusted_aggregate = (
                        adjusted_pair
                        if adjusted_aggregate is None
                        else torch.logaddexp(
                            adjusted_aggregate,
                            adjusted_pair,
                        )
                    )
                    if return_aux:
                        base_aggregate = (
                            pair_score
                            if base_aggregate is None
                            else torch.logaddexp(
                                base_aggregate,
                                pair_score,
                            )
                        )

            num_segment_pairs = self.query_segments * self.key_segments
            scores = adjusted_aggregate - math.log(num_segment_pairs)
            position_features = self._position_features(
                valid_counts.to(device=scores.device),
                key_repr.shape[-3],
                scores.dtype,
            )
            raw_position_residual = torch.einsum(
                "hf,qkf->hqk",
                self.residual_position_weight[layer].to(dtype=compute_dtype),
                position_features,
            ).unsqueeze(0)
            position_residual = (
                0.5
                * self.residual_bound
                * torch.tanh(raw_position_residual)
            )
            scores = scores + position_residual
            if return_aux:
                base_scores = base_aggregate - math.log(num_segment_pairs)
                residual = scores - base_scores
                return scores, {
                    "base_scores": base_scores,
                    "residual": residual,
                    "raw_residual": residual,
                }
            return scores

        if self.bounded_residual and self.streaming_residual:
            aggregate_scores = None
            weighted_pair_scores = None
            pair_weights = self.residual_pair_weight[layer].reshape(
                self.num_heads,
                -1,
            ).to(dtype=compute_dtype)
            pair_weight_index = 0
            if self.residual_use_stats:
                pair_sum = None
                pair_square_sum = None
                pair_max = None
                pair_min = None
            for query_segment in range(self.query_segments):
                for key_segment in range(self.key_segments):
                    pair_score = (
                        torch.einsum(
                            "bhqr,bhkr->bhqk",
                            projected_queries[..., query_segment, :],
                            projected_keys[..., key_segment, :],
                        )
                        / score_divisor
                    )
                    aggregate_scores = (
                        pair_score
                        if aggregate_scores is None
                        else torch.logaddexp(aggregate_scores, pair_score)
                    )
                    weighted_score = pair_score * pair_weights[
                        :, pair_weight_index
                    ].view(1, -1, 1, 1)
                    weighted_pair_scores = (
                        weighted_score
                        if weighted_pair_scores is None
                        else weighted_pair_scores + weighted_score
                    )
                    if self.residual_use_stats:
                        pair_sum = (
                            pair_score
                            if pair_sum is None
                            else pair_sum + pair_score
                        )
                        pair_square = pair_score.square()
                        pair_square_sum = (
                            pair_square
                            if pair_square_sum is None
                            else pair_square_sum + pair_square
                        )
                        pair_max = (
                            pair_score
                            if pair_max is None
                            else torch.maximum(pair_max, pair_score)
                        )
                        pair_min = (
                            pair_score
                            if pair_min is None
                            else torch.minimum(pair_min, pair_score)
                        )
                    pair_weight_index += 1

            num_segment_pairs = self.query_segments * self.key_segments
            base_scores = aggregate_scores - math.log(num_segment_pairs)
            raw_residual = weighted_pair_scores - base_scores * pair_weights.sum(
                dim=-1
            ).view(1, -1, 1, 1)
            if self.residual_use_stats:
                pair_mean = pair_sum / num_segment_pairs
                pair_variance = (
                    pair_square_sum / num_segment_pairs - pair_mean.square()
                ).clamp_min(0.0)
                pair_stats = torch.stack(
                    (
                        pair_variance.sqrt(),
                        pair_max - base_scores,
                        base_scores - pair_min,
                    ),
                    dim=-1,
                ).tanh()
                raw_residual = raw_residual + torch.einsum(
                    "bhqkf,hf->bhqk",
                    pair_stats,
                    self.residual_stat_weight[layer].to(
                        dtype=compute_dtype
                    ),
                )
            position_features = self._position_features(
                valid_counts.to(device=base_scores.device),
                key_repr.shape[-3],
                base_scores.dtype,
            )
            raw_residual = raw_residual + torch.einsum(
                "hf,qkf->hqk",
                self.residual_position_weight[layer].to(dtype=compute_dtype),
                position_features,
            ).unsqueeze(0)
            residual = self.residual_bound * torch.tanh(raw_residual)
            scores = base_scores + residual
            if return_aux:
                return scores, {
                    "base_scores": base_scores,
                    "residual": residual,
                    "raw_residual": raw_residual,
                }
            return scores

        if self.bounded_residual:
            pair_scores = []
            for query_segment in range(self.query_segments):
                for key_segment in range(self.key_segments):
                    pair_scores.append(
                        torch.einsum(
                            "bhqr,bhkr->bhqk",
                            projected_queries[..., query_segment, :],
                            projected_keys[..., key_segment, :],
                        )
                        / score_divisor
                    )
            pair_scores = torch.stack(pair_scores, dim=-1)
            base_scores = torch.logsumexp(pair_scores, dim=-1) - math.log(
                pair_scores.shape[-1]
            )
            centered_pairs = torch.tanh(
                pair_scores - base_scores.unsqueeze(-1)
            )
            pair_weight = self.residual_pair_weight[layer].reshape(
                self.num_heads,
                -1,
            )
            raw_residual = torch.einsum(
                "bhqkp,hp->bhqk",
                centered_pairs,
                pair_weight.to(dtype=compute_dtype),
            )
            if self.residual_rank:
                projected_residual_queries = torch.einsum(
                    "bhqsd,dr->bhqsr",
                    query_compute,
                    self.residual_q_proj[layer].to(dtype=compute_dtype),
                )
                projected_residual_keys = torch.einsum(
                    "bhksd,dr->bhksr",
                    key_compute,
                    self.residual_k_proj[layer].to(dtype=compute_dtype),
                )
                low_rank_pair_scores = []
                for query_segment in range(self.query_segments):
                    for key_segment in range(self.key_segments):
                        low_rank_pair_scores.append(
                            torch.einsum(
                                "bhqr,bhkr->bhqk",
                                projected_residual_queries[
                                    ..., query_segment, :
                                ],
                                projected_residual_keys[
                                    ..., key_segment, :
                                ],
                            )
                            / score_divisor
                        )
                low_rank_features = torch.tanh(
                    torch.stack(low_rank_pair_scores, dim=-1)
                    - pair_scores
                )
                low_rank_weight = self.residual_low_rank_weight[layer].reshape(
                    self.num_heads,
                    -1,
                )
                raw_residual = raw_residual + torch.einsum(
                    "bhqkp,hp->bhqk",
                    low_rank_features,
                    low_rank_weight.to(dtype=compute_dtype),
                )

            pair_stats = torch.stack(
                (
                    pair_scores.std(dim=-1, unbiased=False),
                    pair_scores.amax(dim=-1) - base_scores,
                    base_scores - pair_scores.amin(dim=-1),
                ),
                dim=-1,
            ).tanh()
            raw_residual = raw_residual + torch.einsum(
                "bhqkf,hf->bhqk",
                pair_stats,
                self.residual_stat_weight[layer].to(dtype=compute_dtype),
            )
            position_features = self._position_features(
                valid_counts.to(device=base_scores.device),
                key_repr.shape[-3],
                base_scores.dtype,
            )
            raw_residual = raw_residual + torch.einsum(
                "hf,qkf->hqk",
                self.residual_position_weight[layer].to(dtype=compute_dtype),
                position_features,
            ).unsqueeze(0)
            residual = self.residual_bound * torch.tanh(raw_residual)
            scores = base_scores + residual
            if return_aux:
                return scores, {
                    "base_scores": base_scores,
                    "residual": residual,
                    "raw_residual": raw_residual,
                }
            return scores

        temperature = (
            self.log_temperature[layer]
            .clamp(math.log(0.05), math.log(20.0))
            .exp()
            .to(dtype=compute_dtype)
            .view(1, -1, 1, 1)
        )
        aggregate_scores = None
        for query_segment in range(self.query_segments):
            for key_segment in range(self.key_segments):
                pair_scores = torch.einsum(
                    "bhqr,bhkr->bhqk",
                    projected_queries[..., query_segment, :],
                    projected_keys[..., key_segment, :],
                ) / score_divisor
                pair_scores = pair_scores + self.segment_bias[
                    layer,
                    :,
                    query_segment,
                    key_segment,
                ].view(1, -1, 1, 1)
                scaled_scores = pair_scores / temperature
                aggregate_scores = (
                    scaled_scores
                    if aggregate_scores is None
                    else torch.logaddexp(aggregate_scores, scaled_scores)
                )
        num_segment_pairs = self.query_segments * self.key_segments
        learned_scores = temperature * (
            aggregate_scores - math.log(num_segment_pairs)
        )
        learned_scale = self.log_learned_scale[layer].clamp(-6.0, 6.0).exp()
        scores = learned_scores * learned_scale.view(1, -1, 1, 1)

        if self.use_raw_scores:
            query_mean = query_compute.mean(dim=-2)
            key_mean = key_compute.mean(dim=-2)
            raw_scores = torch.einsum(
                "bhqd,bhkd->bhqk",
                query_mean,
                key_mean,
            ) / math.sqrt(self.head_dim)
            raw_scale = self.log_raw_scale[layer].clamp(-6.0, 6.0).exp()
            scores = scores + raw_scores * raw_scale.view(1, -1, 1, 1)

        position_features = self._position_features(
            valid_counts.to(device=scores.device),
            key_repr.shape[-3],
            scores.dtype,
        )
        position_bias = torch.einsum(
            "hf,qkf->hqk",
            self.position_weight[layer].to(dtype=scores.dtype),
            position_features,
        )
        scores = scores + position_bias.unsqueeze(0)
        if return_aux:
            return scores, {
                "base_scores": scores,
                "residual": torch.zeros_like(scores),
                "raw_residual": torch.zeros_like(scores),
            }
        return scores


def structured_layout(
    query_block_ids: torch.Tensor,
    valid_counts: torch.Tensor,
    num_key_blocks: int,
    density: float,
    q_block_size: int,
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return forced mask, fill candidates, and per-row fill budgets."""
    if query_block_ids.ndim != 1 or valid_counts.ndim != 1:
        raise ValueError("query_block_ids and valid_counts must be vectors")
    if query_block_ids.shape != valid_counts.shape:
        raise ValueError("query_block_ids and valid_counts must have equal shape")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")

    device = query_block_ids.device
    rows = query_block_ids.numel()
    key_positions = torch.arange(num_key_blocks, device=device)
    valid = key_positions.view(1, -1) < valid_counts.view(-1, 1)
    target_k = min(
        num_key_blocks,
        max(1, int(density * num_key_blocks)),
    )
    dense_rows = valid_counts <= target_k
    q_start = torch.div(
        query_block_ids * q_block_size,
        k_block_size,
        rounding_mode="floor",
    ).clamp(max=num_key_blocks - 1)
    recent = (valid_counts - 1).clamp(min=0, max=num_key_blocks - 1)

    reserve_sink = (q_start > 0) & (target_k > 1) & ~dense_rows
    local_budget = target_k - reserve_sink.long()
    local_start = torch.maximum(
        q_start,
        recent - local_budget + 1,
    )
    local = (
        (key_positions.view(1, -1) >= local_start.view(-1, 1))
        & (key_positions.view(1, -1) <= recent.view(-1, 1))
        & valid
    )
    sink = reserve_sink.view(-1, 1) & (key_positions.view(1, -1) == 0)
    forced = torch.where(dense_rows.view(-1, 1), valid, local | sink)
    fill_budget = torch.where(
        dense_rows,
        torch.zeros(rows, device=device, dtype=torch.long),
        (target_k - forced.sum(dim=-1)).clamp_min(0),
    )
    candidates = (
        ~dense_rows.view(-1, 1)
        & valid
        & ~forced
        & (key_positions.view(1, -1) > 0)
        & (key_positions.view(1, -1) < local_start.view(-1, 1))
    )
    return forced, candidates, fill_budget


def fill_topk(
    values: torch.Tensor,
    candidates: torch.Tensor,
    fill_budget: torch.Tensor,
) -> torch.Tensor:
    """Select each row's highest-valued candidates up to its fill budget."""
    if values.ndim != 2 or candidates.shape != values.shape:
        raise ValueError("values and candidates must be equal-shaped matrices")
    selected = torch.zeros_like(candidates)
    if fill_budget.numel() == 0:
        return selected
    effective_budget = torch.minimum(
        fill_budget,
        candidates.sum(dim=-1),
    )
    max_budget = int(effective_budget.max().item())
    if max_budget <= 0:
        return selected
    top_indices = values.masked_fill(
        ~candidates,
        float("-inf"),
    ).topk(max_budget, dim=-1).indices
    keep_by_rank = (
        torch.arange(max_budget, device=values.device).view(1, -1)
        < effective_budget.view(-1, 1)
    )
    selected.scatter_(-1, top_indices, keep_by_rank)
    return selected


def structured_topk_mask(
    scores: torch.Tensor,
    query_block_ids: torch.Tensor,
    valid_counts: torch.Tensor,
    density: float,
    q_block_size: int,
    k_block_size: int,
) -> torch.Tensor:
    """Combine forced structured blocks with score-selected TopK blocks."""
    forced, candidates, fill_budget = structured_layout(
        query_block_ids,
        valid_counts,
        scores.shape[-1],
        density,
        q_block_size,
        k_block_size,
    )
    return forced | fill_topk(scores, candidates, fill_budget)


def oracle_topk_mask(
    target_mass: torch.Tensor,
    valid_counts: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """Build a per-row oracle mask directly from target attention mass."""
    num_key_blocks = target_mass.shape[-1]
    target_k = min(
        num_key_blocks,
        max(1, int(density * num_key_blocks)),
    )
    key_positions = torch.arange(num_key_blocks, device=target_mass.device)
    valid = key_positions.view(1, -1) < valid_counts.view(-1, 1)
    budgets = torch.minimum(
        valid_counts,
        torch.full_like(valid_counts, target_k),
    )
    return fill_topk(target_mass, valid, budgets)


def mask_metrics(
    predicted_mask: torch.Tensor,
    oracle_mask: torch.Tensor,
    target_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure oracle overlap and retained target mass for predicted masks."""
    oracle_counts = oracle_mask.sum(dim=-1).clamp_min(1)
    overlap = (predicted_mask & oracle_mask).sum(dim=-1) / oracle_counts
    retained_mass = (target_mass * predicted_mask).sum(dim=-1)
    return overlap.float(), retained_mass.float()


def allocate_dynamic_row_budgets(
    priorities: torch.Tensor,
    available_counts: torch.Tensor,
    target_k: int,
    *,
    minimum_counts: torch.Tensor | None = None,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
    redistribution_steps: int = 8,
) -> torch.Tensor:
    """Redistribute a static row budget while preserving its exact total."""
    squeeze_batch = priorities.ndim == 2
    if squeeze_batch:
        priorities = priorities.unsqueeze(0)
    if priorities.ndim != 3:
        raise ValueError("priorities must have shape [H,Q] or [B,H,Q]")
    if target_k <= 0:
        raise ValueError("target_k must be positive")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least 1")
    if redistribution_steps <= 0:
        raise ValueError("redistribution_steps must be positive")

    batch_size, num_heads, num_rows = priorities.shape
    available = torch.as_tensor(
        available_counts,
        device=priorities.device,
        dtype=torch.long,
    )
    try:
        available = torch.broadcast_to(
            available,
            (batch_size, num_heads, num_rows),
        )
    except RuntimeError as exc:
        raise ValueError("available_counts cannot broadcast to priorities") from exc

    static_budgets = torch.minimum(
        available,
        torch.full_like(available, int(target_k)),
    )
    lower_target = max(1, int(math.floor(target_k * min_ratio)))
    upper_target = max(target_k, int(math.ceil(target_k * max_ratio)))
    lower = torch.minimum(
        available,
        torch.full_like(available, lower_target),
    )
    if minimum_counts is not None:
        minimum = torch.as_tensor(
            minimum_counts,
            device=priorities.device,
            dtype=torch.long,
        )
        try:
            minimum = torch.broadcast_to(
                minimum,
                (batch_size, num_heads, num_rows),
            )
        except RuntimeError as exc:
            raise ValueError(
                "minimum_counts cannot broadcast to priorities"
            ) from exc
        lower = torch.maximum(lower, torch.minimum(minimum, available))
    upper = torch.maximum(
        lower,
        torch.minimum(
            available,
            torch.full_like(available, upper_target),
        ),
    )

    flat_priority = priorities.float().tanh().reshape(batch_size, -1)
    flat_lower = lower.reshape(batch_size, -1)
    flat_upper = upper.reshape(batch_size, -1)
    target_totals = static_budgets.reshape(batch_size, -1).sum(dim=-1)
    lower_totals = flat_lower.sum(dim=-1)
    upper_totals = flat_upper.sum(dim=-1)
    if not priorities.is_cuda:
        if bool((lower_totals > target_totals).any()):
            raise ValueError("minimum dynamic budgets exceed the static total")
        if bool((upper_totals < target_totals).any()):
            raise ValueError(
                "maximum dynamic budgets cannot reach the static total"
            )

    continuous = flat_lower.float()
    capacity = (flat_upper - flat_lower).float()
    remaining = (target_totals - lower_totals).float()
    for _ in range(redistribution_steps):
        active = capacity > 1e-6
        logits = flat_priority.masked_fill(~active, -1e4)
        weights = torch.softmax(logits, dim=-1) * active
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proposal = remaining.unsqueeze(-1) * weights
        addition = torch.minimum(proposal, capacity)
        continuous = continuous + addition
        capacity = capacity - addition
        remaining = (remaining - addition.sum(dim=-1)).clamp_min(0.0)

    integer = continuous.floor().long()
    integer = torch.minimum(integer, flat_upper)
    remainder = target_totals - integer.sum(dim=-1)
    fractional = continuous - integer.float()
    fractional = fractional.masked_fill(integer >= flat_upper, float("-inf"))
    max_remainder = int(remainder.max().item())
    if max_remainder > 0:
        indices = fractional.topk(max_remainder, dim=-1).indices
        keep = (
            torch.arange(max_remainder, device=priorities.device).view(1, -1)
            < remainder.view(-1, 1)
        )
        increments = torch.zeros_like(integer)
        increments.scatter_(1, indices, keep.long())
        integer = integer + increments

    budgets = integer.view(batch_size, num_heads, num_rows)
    if not priorities.is_cuda and not torch.equal(
        budgets.reshape(batch_size, -1).sum(dim=-1),
        target_totals,
    ):
        raise RuntimeError("dynamic budget rounding did not preserve the total")
    if squeeze_batch:
        budgets = budgets.squeeze(0)
    return budgets


def projected_attention_output(
    output_contrib: torch.Tensor,
    target_mass: torch.Tensor,
    selection: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct projected sparse and Full outputs from block contributions."""
    if output_contrib.shape[:-1] != target_mass.shape:
        raise ValueError("output_contrib and target_mass shapes do not match")
    if selection.shape != target_mass.shape:
        raise ValueError("selection and target_mass shapes do not match")
    gate = selection.to(dtype=output_contrib.dtype)
    selected_mass = (target_mass * gate).sum(dim=-1)
    sparse_output = (
        output_contrib * gate.unsqueeze(-1)
    ).sum(dim=-2) / selected_mass.clamp_min(eps).unsqueeze(-1)
    full_output = output_contrib.sum(dim=-2)
    return sparse_output, full_output, selected_mass


def output_fidelity_metrics(
    output_contrib: torch.Tensor,
    target_mass: torch.Tensor,
    selection: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute relative output error, cosine similarity, and retained mass."""
    sparse_output, full_output, selected_mass = projected_attention_output(
        output_contrib,
        target_mass,
        selection,
        eps=eps,
    )
    error = sparse_output - full_output
    error_energy = error.square().sum(dim=-1)
    full_energy = full_output.square().sum(dim=-1)
    relative_mse = error_energy / full_energy.clamp_min(eps)
    cosine = F.cosine_similarity(
        sparse_output,
        full_output,
        dim=-1,
        eps=eps,
    )
    return {
        "relative_mse": relative_mse,
        "cosine": cosine,
        "selected_mass": selected_mass,
        "full_energy": full_energy,
    }


def output_block_importance(
    output_contrib: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Approximate each block's Full-output error when that block is omitted."""
    if output_contrib.shape[:-1] != target_mass.shape:
        raise ValueError("output_contrib and target_mass shapes do not match")
    full_output = output_contrib.sum(dim=-2)
    omission_delta = (
        target_mass.unsqueeze(-1) * full_output.unsqueeze(-2)
        - output_contrib
    )
    omission_delta = omission_delta / (
        1.0 - target_mass
    ).clamp_min(eps).unsqueeze(-1)
    full_energy = full_output.square().sum(dim=-1, keepdim=True)
    return omission_delta.square().sum(dim=-1) / full_energy.clamp_min(eps)
