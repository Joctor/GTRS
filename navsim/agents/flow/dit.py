import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F

def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm)
        
        self.global_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.local_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        self.scene_query = nn.Parameter(torch.randn(1, 1, dim))

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim * 4, 9 * dim))

    def forward(self, x, t_emb, img_emb, ego_emb, score_emb):
        img_global, _ = self.global_attn(self.scene_query.expand(x.shape[0], -1, -1), img_emb, img_emb)
        img_global = img_global.squeeze(1) # (B, D)
        ego_current = ego_emb[:, -1, :] # (B, D)
        condition = torch.cat([t_emb, img_global, ego_current, score_emb], dim=-1)

        shift_msa, scale_msa, gate_msa, \
        shift_cross, scale_cross, gate_cross, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )

        kv = torch.cat([img_emb, ego_emb], dim=1)
        cross_out, _ = self.local_attn(
                modulate(self.norm_cross(x), scale_cross, shift_cross), 
                kv, kv
            )
        x = x + gate_cross.unsqueeze(1) * cross_out

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.global_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.scene_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size * 4, 2 * hidden_size))

    def forward(self, x, t_emb, img_emb, ego_emb, score_emb):
        img_global, _ = self.global_attn(self.scene_query.expand(x.shape[0], -1, -1), img_emb, img_emb)
        img_global = img_global.squeeze(1) # (B, D)
        ego_current = ego_emb[:, -1, :] # (B, D)
        condition = torch.cat([t_emb, img_global, ego_current, score_emb], dim=-1)

        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


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
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_poses, hidden_size))

        self.score_embed = nn.Sequential(
                nn.Linear(9, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, num_heads, input_size)

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
        nn.init.normal_(self.pos_embed, std=0.02)  # for z_t (8 slots)

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
    def forward(self, z_t, t, img_emb, ego_emb, pdmscore):
        """
        z_t: (B, L=8, D=3)
        t: (B, 1, 1)
        keyval: (B, 64, 256)
        """
        B, L, D = z_t.shape
        zt_emb = self.zt_embed(z_t) + self.pos_embed 

        time_emb = self.t_embed(t.squeeze(-1).squeeze(-1))       # (B, 8, H)

        score_emb = self.score_embed(pdmscore)
        
        x = zt_emb
        # 4. Pass through blocks
        for block in self.blocks:
            x = block(x, time_emb, img_emb, ego_emb, score_emb)
        
        v_t = self.final_layer(x, time_emb, img_emb, ego_emb, score_emb)  # (B, 8, 3) ← only one argument!
        
        return v_t


if __name__ == "__main__":
    B, L, D = 256, 8, 3
    model = MFDiT(input_size=D, num_poses=L, hidden_size=256, depth=2, num_heads=4)

    z_t = torch.randn(B, L, D)              
    t = torch.rand(B, 1, 1)                    
    img_token = torch.randn(B, 64, 256)              
    ego_token = torch.randn(B, 4, 256)
    pdmscore = torch.randn(B, 9)        

    output = model(z_t, t, img_token, ego_token, pdmscore)
    print("Input z_t shape:", z_t.shape)
    print("Output shape:", output[0].shape, output[1].shape)    # Should be (2, 8, 3)