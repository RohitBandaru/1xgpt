import torch
from torch import nn
import os

# Try to import xformers, fall back gracefully if not available
XFORMERS_DISABLED = os.environ.get("XFORMERS_DISABLED", "false").lower() == "true"
try:
    if not XFORMERS_DISABLED:
        from xformers.ops import LowerTriangularMask, memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
    else:
        XFORMERS_AVAILABLE = False
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xformers not available, using BasicSelfAttention fallback")

class BasicSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Scaling by 8 to be equal when head_dim=64
        self.scale = 8/self.head_dim if use_mup else self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.qk_norm = qk_norm
        if self.qk_norm:
            # qk normalization https://arxiv.org/pdf/2302.05442
            # Note that LN is done in fp32, so they have to be
            self.norm = nn.LayerNorm(self.head_dim, eps=1e-05)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)
            # LN done in float32, cast back to bf16
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]            
            mask = ~torch.tril(torch.ones(i, j)).bool().to(attn.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# Only define MemoryEfficientAttention if xformers is available
if XFORMERS_AVAILABLE:
    class MemoryEfficientAttention(BasicSelfAttention):
        # NOTE: Mem-eff attention from xformers is actually Flash Attention 2
            
        def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = unbind(qkv, 2)
            if self.qk_norm:
                q = self.norm(q)
                k = self.norm(k)
                # LN done in float32, cast back to bf16
                q = q.to(dtype=v.dtype)
                k = k.to(dtype=v.dtype)

            attn_bias = LowerTriangularMask() if causal else None
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale)
            x = x.reshape([B, N, C])

            x = self.proj(x)
            return x

    SelfAttention = MemoryEfficientAttention
else:
    SelfAttention = BasicSelfAttention