"""
Temporal detection head for frame-level SSL features (Part 3 back-end).

Input contract:  [B, T, D]  (one SSL layer's frames, from encoders.ssl.encode_frames)
                 or [B, L, T, D] to let the head learn a softmax weighted-sum over layers
                 (SSL-AASIST style — mid layers usually win).
Output:          [B, 2] logits  (human=0, AI=1).

`TemporalAttentionClassifier` is the roadmap's **3c intermediate**: self-attention over frames
+ attentive-statistics readout + MLP. It is stronger than the mean/std linear probe but still
cheap, and it de-risks the jump to heavy heads.

The two heavy heads the mentor wants — **AASIST** (graph attention over a spectral-temporal map,
Jung et al. 2022) and **SpecTTTra** (music transformer, SONICS) — plug into the SAME
[B,T,D]->[B,2] interface and are integrated from their official repos when we train the strong
classifiers *after* the eval is de-confounded (roadmap 3c/3d). They are intentionally NOT
reimplemented from scratch here — a subtly-wrong graph module would be worse than none.

Nothing in this file trains yet; it is scaffolding to be exercised once temporal features are
cached (extract via encode_frames) and the de-confounded eval is ready.
"""
import torch
import torch.nn as nn


class LayerWeightedSum(nn.Module):
    """Learnable softmax weighted-sum over SSL layers: [B, L, T, D] -> [B, T, D]."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):                                  # [B, L, T, D]
        a = torch.softmax(self.w, dim=0)
        return torch.einsum("l,bltd->btd", a, x)


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling over time -> [B, 2*dim] (weighted mean || std)."""

    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x):                                  # [B, T, dim]
        w = torch.softmax(self.att(x), dim=1)              # [B, T, 1]
        mean = (w * x).sum(dim=1)                          # [B, dim]
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        return torch.cat([mean, torch.sqrt(var + 1e-8)], dim=-1)


class TemporalAttentionClassifier(nn.Module):
    """Self-attention over frames + attentive-stats readout + MLP. [B,T,D] (or [B,L,T,D]) -> [B,2]."""

    def __init__(self, in_dim: int, n_ssl_layers: int = None, hid: int = 192,
                 heads: int = 4, depth: int = 2, dropout: float = 0.2, n_classes: int = 2):
        super().__init__()
        self.layer_sum = LayerWeightedSum(n_ssl_layers) if n_ssl_layers else None
        self.proj = nn.Linear(in_dim, hid)
        enc = nn.TransformerEncoderLayer(hid, heads, dim_feedforward=hid * 2,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.pool = AttentiveStatsPool(hid)
        self.cls = nn.Sequential(nn.Linear(hid * 2, hid), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hid, n_classes))

    def forward(self, x):
        if x.dim() == 4:                                   # [B, L, T, D] -> weighted sum
            assert self.layer_sum is not None, "got [B,L,T,D] but n_ssl_layers was not set"
            x = self.layer_sum(x)
        h = self.proj(x)                                   # [B, T, hid]
        h = self.encoder(h)                                # [B, T, hid]
        return self.cls(self.pool(h))                      # [B, 2]
