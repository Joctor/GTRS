import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        # Expand freqs to match t's dims
        t_expanded = t.unsqueeze(-1).float()  # (*t.shape, 1)
        args = t_expanded * freqs             # (*t.shape, half_dim)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        return self.linear(self.norm(x))


class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=3,
        num_poses=8,
        hidden_size=128,
        depth=8,
        num_heads=16,
    ):
        super().__init__()

        self.zt_embed = nn.Linear(input_size, hidden_size)
        
        self.t_embed = TimestepEmbedder(hidden_size)
        
        self.pos_embed = nn.Embedding(num_poses, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, input_size)

        self.initialize_weights()

    def initialize_weights(self):
        # 1. Basic init for all Linear layers (Xavier + zero bias)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. Initialize learnable positional embeddings (hist + future)
        nn.init.normal_(self.pos_embed.weight, std=0.02)  # for z_t (8 slots)

        # 3. Initialize timestep embedding MLP (like DiT)
        # Assuming TimestepEmbedder has: self.mlp = nn.Sequential(Linear, SiLU, Linear)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.constant_(self.t_embed.mlp[0].bias, 0)
        nn.init.constant_(self.t_embed.mlp[2].bias, 0)

        # 4. Zero-out final layer (optional but common in flow/diffusion)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    # (zt | t, bev, ego, score)
    def forward(self, z_t, t, keyval):
        """
        z_t: (B, L=8, D=3)
        t: (B, L)
        egostatus: (B, 4, 11)
        """
        B, L, D = z_t.shape

        # 1. Encode future state tokens (z_t)
        zt_emb = self.zt_embed(z_t)                     # (B, 8, H)
        time_emb = self.t_embed(t)                      # (B, 8, H)
        pos_indices = torch.arange(L, device=z_t.device)  # (L,)
        pos_emb = self.pos_embed(pos_indices).unsqueeze(0)  # (1, L, H)
        future_tokens = zt_emb + time_emb + pos_emb  # (B, 8, H)

        # 3. IN-CONTEXT CONDITIONING: CONCATENATE!
        x = torch.cat([keyval, future_tokens], dim=1)   # (B, 12, H)

        # 4. Pass through blocks
        for block in self.blocks:
            x = block(x)

        # 5. Predict only for future tokens
        x_future = x[:, -L:, :]  # (B, 8, H)
        v_t = self.final_layer(x_future)  # (B, 8, 3) ← only one argument!

        # x_future作为block的输出，传给分数头
        
        return v_t, x_future


if __name__ == "__main__":
    B, L, D = 256, 8, 3
    model = MFDiT(input_size=D, num_poses=L, hidden_size=256, depth=2, num_heads=4)

    z_t = torch.randn(B, L, D)              
    t = torch.rand(B, L)                    
    bev = torch.randn(B, 64 + 4, 256)              
    hist_ego = torch.randn(B, 4, 11)        

    output = model(z_t, t, bev)
    print("Input z_t shape:", z_t.shape)
    print("Output shape:", output[0].shape, output[1].shape)    # Should be (2, 8, 3)