import torch
import torch.nn as nn
from models.layers import CastedLinear, RotaryEmbedding, SwiGLU, RMSNorm, apply_rope

class ModernAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = CastedLinear(d_model, 3 * d_model)
        self.o_proj = CastedLinear(d_model, d_model)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = apply_rope(q, k, cos, sin)
        
        # PyTorch 2.0 SDPA
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, L, D))

class Block(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = ModernAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class ModernPendulumTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.model_dim
        
        # ✅ CNN Backbone: 压缩率还是 64 倍。 36000 压缩后序列长度约为 562
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, 7, 4, 3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 7, 4, 3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, self.d_model, 7, 4, 3), nn.BatchNorm1d(self.d_model), nn.GELU(),
        )
        
        # RoPE & Layers
        self.rope = RotaryEmbedding(self.d_model // cfg.num_heads)
        self.layers = nn.ModuleList([
            Block(self.d_model, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        
        # Output Head
        self.pool_attn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), nn.Tanh(),
            nn.Linear(self.d_model // 2, 1), nn.Softmax(dim=1)
        )
        self.final_norm = RMSNorm(self.d_model)
        self.head = nn.Sequential(
            CastedLinear(self.d_model, self.d_model), nn.GELU(),
            CastedLinear(self.d_model, cfg.output_dim)
        )
        
    def forward(self, x):
        x = self.backbone(x.transpose(1, 2)).transpose(1, 2)
        B, L, D = x.shape
        cos, sin = self.rope(L, x.device)
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        x = self.final_norm(x)
        w = self.pool_attn(x)
        x_pool = torch.sum(x * w, dim=1)
        return self.head(x_pool)