"""
saint_model.py
──────────────
SAINT: Self-Attention and Intersample Attention Transformer for tabular data.
Paper: Somepalli et al., 2021  (https://arxiv.org/abs/2106.01342)

Input features split:
  Categorical (survey)   → per-feature embedding tables  → (B, n_cat,  dim)
  Continuous  (EEG+Watch+Age) → per-feature linear proj  → (B, n_cont, dim)
  Combined               → (B, n_features + 1, dim)  [+1 for CLS token]

Two-attention SAINT block:
  1. Row-wise    — each sample attends across its own features
  2. Column-wise — each feature attends across all samples in the batch
                   (SAINT's key innovation for tabular data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, D))


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class SAINTBlock(nn.Module):
    """Row-wise then column-wise self-attention."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1_r = nn.LayerNorm(dim); self.attn_r = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2_r = nn.LayerNorm(dim); self.ffn_r  = FFN(dim, mlp_ratio, dropout)
        self.norm1_c = nn.LayerNorm(dim); self.attn_c = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2_c = nn.LayerNorm(dim); self.ffn_c  = FFN(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Row attention: across features within each sample
        x = x + self.attn_r(self.norm1_r(x))
        x = x + self.ffn_r(self.norm2_r(x))
        # Column attention: across samples for each feature position
        x_t = x.permute(1, 0, 2)                        # (N, B, D)
        x_t = x_t + self.attn_c(self.norm1_c(x_t))
        x_t = x_t + self.ffn_c(self.norm2_c(x_t))
        return x_t.permute(1, 0, 2)                     # (B, N, D)


class SAINT(nn.Module):
    """
    SAINT classifier for multimodal tabular data (Survey + EEG + Watch).

    Args:
        cat_dims  : vocab size per categorical feature (for embedding tables)
        n_cont    : number of continuous features (EEG agg + Watch agg + Age)
        dim       : embedding dimension
        depth     : number of SAINT blocks
        n_heads   : attention heads
        n_classes : output classes (3: low/medium/high pain)
    """

    def __init__(
        self,
        cat_dims:        List[int],
        n_cont:          int,
        dim:             int   = 32,
        depth:           int   = 3,
        n_heads:         int   = 4,
        n_classes:       int   = 3,
        mlp_ratio:       int   = 4,
        dropout:         float = 0.1,
        mlp_hidden_mult: int   = 4,
    ):
        super().__init__()
        self.n_cat  = len(cat_dims)
        self.n_cont = n_cont
        self.dim    = dim

        # Categorical embeddings — one table per feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(v + 1, dim) for v in cat_dims   # +1 for unknown
        ])
        # Continuous projections — one linear per feature
        self.cont_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(1, dim), nn.ReLU())
            for _ in range(n_cont)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            SAINTBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # Classification head (operates on CLS token output)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_hidden_mult, n_classes),
        )

        # Contrastive projection head (pre-training only)
        n_feat = self.n_cat + n_cont
        self.proj_head = nn.Sequential(
            nn.Linear(dim * (n_feat + 1), 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

        self._init_weights()

    def _init_weights(self):
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        for proj in self.cont_projections:
            nn.init.xavier_uniform_(proj[0].weight)

    def embed(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """Build token sequence: [CLS, cat_tokens..., cont_tokens...]"""
        tokens = [self.cls_token.expand(x_cat.size(0), -1, -1)]
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))
        for i, proj in enumerate(self.cont_projections):
            tokens.append(proj(x_cont[:, i:i+1]).unsqueeze(1))
        return torch.cat(tokens, dim=1)   # (B, n_feat+1, dim)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x   = self.embed(x_cat, x_cont)
        x   = self.blocks(x)
        return self.head(x[:, 0])         # classify from CLS token

    def encode(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """Flattened representation for contrastive pre-training."""
        x = self.embed(x_cat, x_cont)
        x = self.blocks(x)
        return self.proj_head(x.flatten(1))


# ── Contrastive utilities ─────────────────────────────────────────────────────

def corrupt_batch(
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
    corruption_rate: float = 0.3,
) -> tuple:
    """SCARF-style corruption: replace features with values from random other rows."""
    B = x_cat.size(0)
    def corrupt(x):
        mask     = torch.rand_like(x.float()) < corruption_rate
        shuffled = x[torch.randint(0, B, (B,), device=x.device)]
        return torch.where(mask, shuffled, x)
    return corrupt(x_cat), corrupt(x_cont)


class ContrastiveLoss(nn.Module):
    """NT-Xent loss between original and corrupted embeddings."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B  = z1.size(0)
        z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2B, d)
        sim = (z @ z.T) / self.T
        sim.masked_fill_(torch.eye(2 * B, device=z.device).bool(), float("-inf"))
        labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                             torch.arange(0, B,   device=z.device)])
        return F.cross_entropy(sim, labels)