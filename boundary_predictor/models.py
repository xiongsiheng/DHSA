"""
Supported boundary prediction architectures:
  - BoundarySimilarity, BoundarySimilarityAttn, BoundarySimilarityStack.
"""
import torch
from torch import nn
import torch.nn.functional as F



def _unfold_windows(x, window_size):
    """
    Unfold windows of size 2*window_size from the input sequence.

    Args:
    x : (B, C, L)   token embeddings (C = H·D)
    window_size : (int)      window size (2*window_size)

    returns (B, L-1, 2*window_size, C)
    """
    # pad left & right by window_size-1 & window_size tokens
    x = F.pad(x, (window_size-1, window_size), value=0.0)    # (B, C, L+2*window_size-1)
    # unfold: sliding blocks of length 2*window_size along the last dim
    blocks = x.unfold(dimension=2, size=2*window_size, step=1)  # (B,C,L,2*window_size)
    return blocks.permute(0, 2, 3, 1).contiguous()  # (B,L,2*window_size,C)


class BoundarySimilarity(nn.Module):
    """
    A boundary predictor that encodes two windows with CNN
    and predicts the similarity between them.
    """
    def __init__(self, channel_in, window_size=4, d_h=256, d_in=256):
        """
        Args:
            channel_in: Input channel dimension.
            window_size: Window size.
            d_h: Hidden dimension.
            d_in: Input dimension.
        """
        super().__init__()
        self.window_size = window_size

        # 1. shared “window encoder”
        self.enc = nn.Sequential(
            nn.Conv1d(channel_in, channel_in,
                      kernel_size=window_size,
                      groups=channel_in),  # mix positions
            nn.ReLU(),
            nn.Conv1d(channel_in, d_h, kernel_size=1),  # mix channels
            nn.ReLU()
        )

        # 2. final decision MLP  (input: 4·d_h + 1)
        self.mlp = nn.Sequential(
            nn.Linear(4*d_h + 1, d_in),
            nn.ReLU(),
            nn.Linear(d_in, 1)
        )

    def forward(self, x):
        # input x: (B, C, L)
        # ----- extract 2*window_size windows for every gap -----
        blocks = _unfold_windows(x, self.window_size)  # (B, L, 2*window_size, C)
        left = blocks[:, :, :self.window_size].permute(0, 1, 3, 2)   # (B,L,C,window_size)
        right = blocks[:, :, self.window_size:].permute(0, 1, 3, 2)   # (B,L,C,window_size)

        # ----- encode windows -----
        b, l, c, _ = left.shape
        left_enc = self.enc(left.reshape(-1, c, self.window_size))
        left_enc = left_enc.squeeze(-1)   # (B*L, d_h)
        right_enc = self.enc(right.reshape(-1, c, self.window_size))
        right_enc = right_enc.squeeze(-1)  # (B*L, d_h)
        left_enc = left_enc.view(b, l, -1)
        right_enc = right_enc.view(b, l, -1)

        # ----- similarity & feature fusion -----
        sim = F.cosine_similarity(left_enc, right_enc, dim=-1, eps=1e-6)
        sim = sim.unsqueeze(-1)  # (B,L,1)
        feat = torch.cat((
            left_enc,
            right_enc,
            torch.abs(left_enc - right_enc),
            left_enc * right_enc,
            sim
        ), dim=-1)                                               # (B,L,4d_h+1)

        logits = self.mlp(feat).squeeze(-1)                      # (B,L)
        return logits


class WindowAttn(nn.Module):
    """
    An encoder that encodes a window using multi-head attention.
    """
    def __init__(self, channel_in, num_heads=4):
        """
        Args:
            channel_in: Input channel dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(channel_in, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channel_in)

    def forward(self, window):
        # input window: (B*L, 2*window_size, C)
        # Standard self‑attention over the 2*window_size tokens
        attn_out, _ = self.mha(window, window, window, need_weights=False)
        return self.norm(attn_out)            # residual optional


class WindowPool(nn.Module):
    """
    A pooling layer that pools over a window.
    """
    def __init__(self, channel_in):
        """
        Args:
            channel_in: Input channel dimension.
        """
        super().__init__()
        self.gate = nn.Linear(channel_in, 1, bias=False)    # (C)→1 score

    def forward(self, tokens):
        # input tokens: (B*L, window_size, C)
        # scoring: shape -> (B*L, window_size, 1)
        score = self.gate(tokens)
        weights = torch.softmax(score, dim=1)         # across window_size tokens
        return (weights * tokens).sum(dim=1)          # (B*L, C)


class BoundarySimilarityAttn(nn.Module):
    """
    A boundary predictor that encodes two windows with multi-head attention
    and predicts the similarity between them.
    """
    def __init__(
        self,
        channel_in,
        window_size=4,
        d_h=256,
        heads=4,
        window_pool=False
    ):
        """
        Args:
            channel_in: Input channel dimension.
            window_size: Window size.
            d_h: Hidden dimension.
            heads: Number of attention heads.
            window_pool: Whether to use window pooling.
        """
        super().__init__()
        self.window_size = window_size
        self.enc = WindowAttn(channel_in, heads)
        self.pool = WindowPool(channel_in) if window_pool else None

        self.mlp = nn.Sequential(
            nn.Linear(4*channel_in + 1, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1)
        )

    def forward(self, x):
        # input x: (B, C, L)
        blocks = _unfold_windows(x, self.window_size)          # (B,L,2*window_size,C)
        b, l, _, c = blocks.shape
        flat = blocks.view(-1, 2*self.window_size, c)         # (B*L,2*window_size,C)
        enc = self.enc(flat)                       # (B*L, 2*window_size, C)
        if self.pool is None:
            left_enc  = enc[:, :self.window_size].mean(dim=1)    # average over first window_size tokens  → (B*L, C)
            right_enc = enc[:, self.window_size:].mean(dim=1)    # average over last window_size tokens  → (B*L, C)
        else:
            left_enc  = self.pool(enc[:, :self.window_size])    # first window_size tokens
            right_enc = self.pool(enc[:, self.window_size:])    # last window_size tokens
        left_enc = left_enc.view(b, l, c)
        right_enc = right_enc.view(b, l, c)
        sim = F.cosine_similarity(left_enc, right_enc, dim=-1, eps=1e-6)
        sim = sim.unsqueeze(-1)
        feat = torch.cat((left_enc, right_enc,
                          torch.abs(left_enc - right_enc),
                          left_enc * right_enc, sim), dim=-1)   # (B,L, 4·C + 1)
        return self.mlp(feat).squeeze(-1)


class BoundarySimilarityStack(nn.Module):
    """
    A boundary predictor that encodes two windows with stacked multi-head attention
    and predicts the similarity between them.
    """
    def __init__(
        self,
        channel_in,
        n_blocks=4,
        window_size=4,
        d_h=256,
        heads=4,
        window_pool=False,
        p_drop=0.1
    ):
        """
        Args:
            channel_in: Input channel dimension.
            n_blocks: Number of attention blocks.
            window_size: Window size.
            d_h: Hidden dimension.
            heads: Number of attention heads.
            window_pool: Whether to use window pooling.
            p_drop: Dropout probability.
        """
        super().__init__()
        self.window_size = window_size
        self.enc_ls = nn.ModuleList([WindowAttn(channel_in, heads) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(p_drop)
        self.norm = nn.LayerNorm(channel_in)      # extra norm
        self.pool = WindowPool(channel_in) if window_pool else None
        self.mlp = nn.Sequential(
            nn.Linear(4*channel_in + 1, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1)
        )

    def forward(self, x):
        # input x: (B, L, C)
        blocks = _unfold_windows(x, self.window_size)           # (B, L, 2*window_size, C)
        b, l, _, c = blocks.shape
        x = blocks.reshape(-1, 2*self.window_size, c)          # (B*L, 2*window_size, C)

        for enc in self.enc_ls:
            x = x + self.dropout(enc(self.norm(x)))  # Pre‑LN residual

        # ---- pool & classify (unchanged) ----
        if self.pool is None:
            left = x[:, :self.window_size].mean(dim=1)
            right = x[:, self.window_size:].mean(dim=1)
        else:
            left = self.pool(x[:, :self.window_size])
            right = self.pool(x[:, self.window_size:])

        left, right = left.view(b, l, c), right.view(b, l, c)
        sim = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
        feat = torch.cat([left, right, torch.abs(left - right), left * right, sim], dim=-1)
        return self.mlp(feat).squeeze(-1)  # (B, L) logits