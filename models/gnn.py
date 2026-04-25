import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerScale(nn.Module):
    """LayerScale (common in modern Transformers/GNNs for stable deep training)."""
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class AttentionPooling(nn.Module):
    """Unchanged but with better init."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.score.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=1)
        return (w * h).sum(dim=1)


class GATv2Layer(nn.Module):
    """Improved GATv2 with soft adjacency support (add log(A) to scores).
    Follows GATv2 spirit + recent EEG-GNN practices.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        share_weights: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = self.W_src if share_weights else nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention vector (additive attention style from GATv2)
        self.att_vec = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.layer_scale = LayerScale(out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_src.weight)
        if not hasattr(self, "W_dst") or self.W_dst is not self.W_src:
            nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.att_vec.view(1, -1).unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.heads, self.head_dim

        src = self.W_src(x).view(B, N, H, D)
        dst = self.W_dst(x).view(B, N, H, D)

        # Additive attention (GATv2 style)
        e = F.leaky_relu(
            src.unsqueeze(2) + dst.unsqueeze(1), negative_slope=self.negative_slope
        )
        score = (e * self.att_vec).sum(dim=-1)  # (B, N, N, H)

        # Soft adjacency integration (SOTA practice instead of hard mask)
        A = A.unsqueeze(-1).expand(-1, -1, -1, H)  # (B, N, N, H)
        score = score + torch.log(A.clamp(min=1e-8))

        alpha = F.softmax(score, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum("bijh,bjhd->bihd", alpha, dst)
        out = out.reshape(B, N, -1) + self.bias
        return self.layer_scale(out)


class DynamicGraphLearner(nn.Module):
    """Core contribution: Dynamic & Learnable Adjacency (2024-2025 SOTA).
    
    Combines:
      1. Learnable node embeddings (global structure prior).
      2. Input-dependent dynamic similarity from current node features h.
      3. Learnable blending with parametric prior.
      4. Top-K sparsity, symmetrization, self-loops, renormalization.
    
    Inspired by DGAT, DSSTNet, FreqDGT, STGATE, AGGCN, etc.
    """
    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 64,
        top_k: int = 8,
        tau: float = 0.5,
        use_prior: bool = True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.top_k = top_k
        self.tau = nn.Parameter(torch.tensor(tau))

        self.node_emb = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.02)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)  # for dynamic part

        self.prior_weight = nn.Parameter(torch.tensor(0.3))  # blending factor
        self.prior_adj = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01) if use_prior else None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.prior_adj is not None:
            nn.init.xavier_uniform_(self.prior_adj)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D) -> dynamic A: (B, N, N)"""
        B, N, _ = h.shape

        # 1. Global structure from learnable node embeddings
        emb_sim = torch.matmul(self.node_emb, self.node_emb.transpose(0, 1)) / self.tau
        A_global = F.softmax(emb_sim, dim=-1).unsqueeze(0).expand(B, -1, -1)

        # 2. Dynamic per-sample similarity from current features
        h_proj = self.proj(h)
        dyn_sim = torch.matmul(h_proj, h_proj.transpose(-2, -1)) / h_proj.size(-1)**0.5
        A_dyn = F.softmax(dyn_sim / self.tau, dim=-1)

        A = 0.6 * A_global + 0.4 * A_dyn

        # 3. Optional parametric prior blending
        if self.prior_adj is not None:
            prior = (self.prior_adj + self.prior_adj.transpose(0, 1)) / 2
            prior = torch.sigmoid(prior).unsqueeze(0).expand(B, -1, -1)
            alpha = torch.sigmoid(self.prior_weight)
            A = alpha * prior + (1 - alpha) * A

        # 4. Post-processing (standard in recent papers)
        A = (A + A.transpose(-2, -1)) / 2.0                                 # symmetric
        A = A + torch.eye(N, device=A.device).unsqueeze(0) * 0.2           # self-loops
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)                       # renormalization

        # 5. Top-K sparsity (highly recommended in EEG-GNN literature)
        if self.top_k is not None and self.top_k < N:
            topk_val, _ = torch.topk(A, self.top_k, dim=-1)
            threshold = topk_val[..., -1:].clone()
            mask = (A >= threshold).to(A.dtype)
            A = A * mask
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        return A


class NodeEncoder(nn.Module):
    """Improved with LayerScale, better residuals, and modern init."""
    def __init__(
        self,
        samples: int = 2000,
        node_hidden: int = 48,
        patch: int = 10,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert samples % patch == 0
        self.patch = patch
        self.patch_size = samples // patch
        D = node_hidden

        # Multi-scale conv stem
        self.conv_stem = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
            for k in (3, 7, 15)
        ])
        self.stem_ln = nn.LayerNorm(samples)

        # Patch embedding
        self.patch_embed = nn.Conv1d(1, D, kernel_size=self.patch_size,
                                     stride=self.patch_size, bias=True)
        self.patch_pos = nn.Parameter(torch.randn(1, patch, D) * 0.02)

        # Cross-frequency Transformer (Pre-LN)
        self.attn_ln = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(
            D, heads, dropout=dropout, batch_first=True, bias=True
        )
        self.attn_scale = LayerScale(D)

        self.ffn_ln = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * 4, D),
            nn.Dropout(dropout),
        )
        self.ffn_scale = LayerScale(D)

        self.out_ln = nn.LayerNorm(D)
        self.pool = AttentionPooling(D)

        self._init_weights()

    def _init_weights(self):
        for conv in self.conv_stem:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="linear")
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out", nonlinearity="linear")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, F = x.shape
        BN = B * N
        xn = x.reshape(BN, 1, F)

        # Multi-scale conv stem + residual
        x_res = x.reshape(BN, F)
        for conv in self.conv_stem:
            x_res = x_res + conv(xn).squeeze(1)
        x_enh = self.stem_ln(x_res).reshape(B, N, F)

        # Patch embedding + position
        tokens = self.patch_embed(x_enh.reshape(BN, 1, F)).transpose(1, 2)  # (BN, patch, D)
        tokens = tokens + self.patch_pos

        # Transformer block (Pre-LN + residual + LayerScale)
        h = self.attn_ln(tokens)
        attn_out, _ = self.attn(h, h, h)
        tokens = tokens + self.attn_scale(attn_out)

        ffn_out = self.ffn(self.ffn_ln(tokens))
        tokens = tokens + self.ffn_scale(ffn_out)

        tokens = self.out_ln(tokens)
        return self.pool(tokens).reshape(B, N, -1)  # (B, N, D)


class GNN(nn.Module):
    """Complete model with dynamic learnable adjacency (SOTA design)."""
    def __init__(
        self,
        n_classes: int = 2,
        samples: int = 2000,
        patch: int = 10,
        n_nodes: int = 30,
        node_hidden: int = 48,
        gcn_hidden: int = 96,
        heads: int = 4,
        dropout: float = 0.1,
        drop_edge: float = 0.1,
        graph_topk: int = 8,
    ):
        super().__init__()
        self.drop_edge = drop_edge
        self.n_nodes = n_nodes

        self.node_encoder = NodeEncoder(
            samples=samples, patch=patch, node_hidden=node_hidden,
            heads=heads, dropout=dropout
        )

        self.graph_learner = DynamicGraphLearner(
            n_nodes=n_nodes, embed_dim=node_hidden, top_k=graph_topk
        )

        # Two GATv2 layers (deeper modeling, common in recent papers)
        self.gat1 = GATv2Layer(
            in_dim=node_hidden, out_dim=gcn_hidden, heads=heads,
            dropout=dropout, share_weights=True
        )

        self.ln1 = nn.LayerNorm(gcn_hidden)
        self.drop = nn.Dropout(dropout)

        self.skip1 = nn.Linear(node_hidden, gcn_hidden, bias=False) if node_hidden != gcn_hidden else nn.Identity()

        self.pool = AttentionPooling(gcn_hidden)
        self.emotion_head = nn.Linear(gcn_hidden, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.emotion_head.weight)
        if self.emotion_head.bias is not None:
            nn.init.zeros_(self.emotion_head.bias)

    def _drop_edge(self, A: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_edge <= 0.0:
            return A
        mask = torch.bernoulli(torch.full_like(A, 1.0 - self.drop_edge))
        mask = mask * (mask.transpose(-2, -1))  # keep symmetry
        eye = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        return A * mask + eye * A * (1 - mask)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(x)                    # (B, N, node_hidden)
        A = self.graph_learner(h)                   # dynamic & learnable
        A = self._drop_edge(A)

        h1 = self.gat1(h, A)
        h1 = self.drop(F.gelu(self.ln1(h1)))
        h1 = h1 + self.skip1(h)

        return self.pool(h1)                        # (B, gcn_hidden)

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        emotion_logit = self.emotion_head(feat)
        return emotion_logit, feat