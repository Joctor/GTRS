import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.flow.flow_config import FlowConfig
from navsim.agents.flow.dit import MFDiT

from .layers.image_encoder.dinov2_lora import ImgEncoder


class FlowHead(nn.Module):
    def __init__(self, num_poses: int, 
                 d_ffn: int, d_model: int,
                 nhead: int, nlayers: int, config: FlowConfig = None
                 ):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        self.num_proposals = config.num_proposals

        self.mfdit = MFDiT(input_size=3, 
              num_poses=num_poses, 
              hidden_size=d_model, 
              depth=nlayers, 
              num_heads=nhead)

    # 把生成的轨迹拼接上bev特征，送入分数头，得到pdm_score，再选出最高的轨迹
    def forward(self, kv: torch.Tensor):
        # bev_token + hist_ego + cur_ego + pos_emd
        B = kv.shape[0]

        # 1. 扩展 Batch
        kv_expanded = kv.repeat_interleave(self.num_proposals, dim=0)
        
        # 2. 初始化状态：直接代表未来 8 个点 (t=1...8)
        future_traj = torch.randn(B * self.num_proposals, self.num_poses, 3, device=kv.device)

        segment_dt = 1.0 / self.num_poses

        with torch.no_grad():
            # s = 0 (因为只有一步，起始点就是唯一的查询点)
            s = 0.0
            
            # --- 构造输入 ---
            # 完整序列 [0, x_1, ..., x_8]
            zeros = torch.zeros(B * self.num_proposals, 1, 3, device=kv.device)
            full_traj = torch.cat([zeros, future_traj], dim=1)
            
            # 起点序列 [x_0, x_1, ..., x_7]
            starts = full_traj[:, :-1, :]
            
            # 【关键】：当 s=0 时，z_t_input = starts
            # 模型将基于“当前起点”预测“直达终点的速度”
            z_t_input = starts 
            
            # 时间输入：每段的起始时间 [0, 1/8, 2/8, ..., 7/8]
            segment_bases = torch.arange(self.num_poses, device=kv.device).float() * segment_dt
            t_input = segment_bases.unsqueeze(0).expand(B * self.num_proposals, -1)
            
            # --- 模型预测 ---
            # 模型看到起点，直接输出恒定速度 v ≈ (x_end - x_start) / dt
            v_t, x_future = self.mfdit(z_t_input, t_input, kv_expanded)
            
            # 计算新的终点估计
            new_ends = starts + v_t * segment_dt
            
            # 更新 future_traj (它代表 ends)
            future_traj = new_ends

        # 3. 恢复形状
        trajectory_proposals = future_traj.view(B, self.num_proposals, self.num_poses, 3)

        return trajectory_proposals, x_future
    
    def get_flow_loss(self, kv, gt_trajectory):
        # kv: (batch_size, num_bev_tokens + 4, config.tf_d_model)
        # gt_trajectory: (batch_size, 8, 3)
        B = kv.shape[0]
        device = kv.device
        gt_trajectory = gt_trajectory.float()
        delta_t = 1 / self.num_poses

        zeros = torch.zeros_like(gt_trajectory[:, :1, :], device=device)  # (B, 1, 3)
        gt_trajectory = torch.cat([zeros, gt_trajectory], dim=1)

        # 2. 为每一段采样一个时间点 → (B, 8)
        local_t = torch.rand(B, self.num_poses, device=device)  # ~ U(0,1)
        segment_start = torch.arange(self.num_poses, device=device).float() * delta_t  # (8,)
        global_t = segment_start + local_t * delta_t  # (B, 8)

        # 3. 提取每段的起点和终点（无需循环！）
        x_k  = gt_trajectory[:, :-1, :]   # (B, 8, 3) —— indices 0 to 7
        x_k1 = gt_trajectory[:, 1:,  :]   # (B, 8, 3) —— indices 1 to 8

        # 4. 插值得到 z_t
        z_t = (1 - local_t.unsqueeze(-1)) * x_k + local_t.unsqueeze(-1) * x_k1  # (B, 8, 3)
        # z_t = z_t + 0.01 * torch.randn_like(z_t)  # 加噪声

        # 5. 真实条件速度场（每段恒定）
        cvf = (x_k1 - x_k) / delta_t  # (B, 8, 3)

        # 6. 模型预测 dit
        v_t, _ = self.mfdit(z_t, global_t, kv)

        return F.mse_loss(v_t, cvf)

class ScoreHead(nn.Module):
    def __init__(self, num_poses: int, 
                 d_ffn: int, d_model: int,
                 nhead: int, nlayers: int, config: FlowConfig = None
                 ):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        self.num_proposals = config.num_proposals

        scorer_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ffn, 
            dropout=0.0, batch_first=True
        )
        self.scorer_refine_attn = nn.TransformerDecoder(scorer_layer, num_layers=nlayers)

        self.head_multipliers = self._build_head(d_model, d_ffn, 4)  # NC, DAC, DDC, TLC
        self.head_weighted = self._build_head(d_model, d_ffn, 5)     # TTC, EP, LK, HC, EC

    def _build_head(self, d_in, d_hidden, d_out):
        return nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, trajectories: torch.Tensor, trajectory_feature: torch.Tensor, 
                kv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectories: (B, K, L, 3)
            trajectory_feature: (B*K, L, D) 来自 FlowHead
            kv: (B, N, D)
        """
        B, K, L, _ = trajectories.shape
        
        # 1. 扩展 KV 和 Status
        kv_expanded = kv.repeat_interleave(K, dim=0) # (B*K, N, D)
        
        # 2. 二次环境查询 (Refine)
        # tgt: (B*K, L, D), memory: (B*K, N, D)
        refined_features = self.scorer_refine_attn(tgt=trajectory_feature, memory=kv_expanded)

        logits_mult = self.head_multipliers(refined_features.mean(dim=1)) 
        logits_weighted = self.head_weighted(refined_features.mean(dim=1))

        # 恢复形状为 (B, K, ...)
        pred_mult = torch.sigmoid(logits_mult).view(B, K, 4) # (B, K, 4)
        pred_weighted = torch.sigmoid(logits_weighted).view(B, K, 5) # (B, K, 5)
        
        # --- 定义权重 (根据 NAVSIM v2 EPDMS) ---
        # Multipliers 不参与加权求和，而是作为连乘因子
        # Weighted 指标权重
        w_ttc, w_ep, w_lk, w_hc, w_ec = 5.0, 5.0, 2.0, 2.0, 2.0
        sum_weights = w_ttc + w_ep + w_lk + w_hc + w_ec
        
        # 计算 Weighted Sum (归一化)
        weighted_sum = (
            w_ttc  * pred_weighted[:, :, 0] +
            w_ep   * pred_weighted[:, :, 1] +
            w_lk   * pred_weighted[:, :, 2] +
            w_hc   * pred_weighted[:, :, 3] +
            w_ec   * pred_weighted[:, :, 4]
        ) / sum_weights # (B, K)
        
        # 计算 Multiplier Product (连乘)
        # NC (idx 0), DAC (idx 1), DDC (idx 2), TLC (idx 3)
        multiplier_product = (
            pred_mult[:, :, 0] * 
            pred_mult[:, :, 1] * 
            pred_mult[:, :, 2] * 
            pred_mult[:, :, 3]
        ) # (B, K)
        
        # 最终 EPDMS 分数
        epdms_score = multiplier_product * weighted_sum # (B, K)
        best_idx = torch.argmax(epdms_score, dim=1, keepdim=True) # (B, 1)
        
        # Gather 最佳轨迹
        idx_expanded = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, 3)
        best_trajectory = torch.gather(trajectories, 1, idx_expanded).squeeze(1) # (B, L, 3)
        
        # 准备返回字典
        result = {
            'trajectory': best_trajectory,                  # (B, L, 3) 最终输出
            'all_trajectories': trajectories,               # (B, K, L, 3) 所有候选
            'epdms_score': epdms_score,                     # (B, K) 总分
            'pred_mult': pred_mult,                # (B, K, 4) 各项安全分
            'pred_weighted': pred_weighted,         # (B, K, 5) 各项质量分
            'best_idx': best_idx                            # (B, 1) 最佳索引
        }
        
        return result

class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self._config = config
        self.image_backbone = ImgEncoder(config)

        self.scene_embeds = nn.Parameter(torch.randn(1, self._config.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6)
        
        kv_len = self._config.num_cams * self._config.num_scene_tokens
        if self._config.use_hist_ego_status:
            kv_len += self._config.hist_ego_len
            self.hist_encoding = nn.Linear(config.hist_ego_dim, config.tf_d_model)
        
        self._keyval_embedding = nn.Embedding(
            kv_len, config.tf_d_model
        )  # 8x8 feature grid + trajectory

        self._trajectory_head = FlowHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            config=config
        )

        self._score_head = ScoreHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            config=config
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # image, bev, agent, traj, score
        camera_feature: torch.Tensor = features["image"]
        status_feature: torch.Tensor = features["ego_status"]
        batch_size = status_feature.shape[0]

        scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)

        img_tokens = self.image_backbone(camera_feature, scene_tokens)
        # img_tokens = torch.randn(batch_size, 
        #                         self._config.num_cams * self._config.num_scene_tokens, 
        #                         self._config.tf_d_model)
        
        if self._config.use_hist_ego_status:
            ego_token = self.hist_encoding(status_feature)
            keyval = torch.concatenate([img_tokens, ego_token], dim=1)

        keyval += self._keyval_embedding.weight[None, ...]
        
        trajectory_proposals, trajectory_feature = self._trajectory_head(keyval)

        output = self._score_head(trajectory_proposals, trajectory_feature, keyval)

        # (batch_size, num_bev_tokens + 4, config.tf_d_model)
        output['env_kv'] = keyval

        return output

