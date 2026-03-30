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
        self.state_size = config.state_size
        self.num_proposals = config.num_proposals

        self.mfdit = MFDiT(input_size=config.state_size, 
              num_poses=num_poses, 
              hidden_size=d_model, 
              depth=nlayers, 
              num_heads=nhead)

    def forward(self, keyval):
        # bev_token + pos_emd
        B = keyval.shape[0]
        device = keyval.device

        with torch.no_grad():
            x_t = torch.randn(B, self.num_poses, self.state_size, device=device)  # (B, 8, 3)
            num_steps = 10 
            dt = 1.0 / num_steps

            for i in range(num_steps):
                # 构造时间步 t，从 0 到 1-dt
                t_val = torch.full((B, 1, 1), i * dt, device=device)
                
                # 预测速度
                v_t = self.mfdit(x_t, t_val, keyval)
                
                # 更新: x_{t+dt} = x_t + v_t * dt
                x_t = x_t + v_t * dt

            trajectory_proposals = x_t.view(-1, self.num_proposals, self.num_poses, self.state_size)
            
            return trajectory_proposals
    
    def get_flow_loss(self, keyval, gt_trajectory):
        # kv: (batch_size, num_bev_tokens + 4, config.tf_d_model)
        # gt_trajectory: (batch_size, 8, 3)
        B = keyval.shape[0]
        device = keyval.device
        x_1 = gt_trajectory.repeat_interleave(self.num_proposals, dim=0)  # (B, 8, 3)
        x_0 = torch.randn_like(x_1)  # (B, 8, 3)
        t = torch.rand(B, 1, 1, device=device)

        # 4. 插值得到 z_t
        z_t = x_0 + t * (x_1 - x_0)  # (B, 8, 3)
        # z_t = z_t + 0.01 * torch.randn_like(z_t)  # 加噪声

        # 5. 真实条件速度场（每段恒定）
        cvf = (x_1 - x_0)  # (B, 8, 3)

        # 6. 模型预测 dit
        v_t = self.mfdit(z_t, t, keyval)

        return F.mse_loss(v_t, cvf)

class ScoreHead(nn.Module):
    def __init__(self, num_poses: int, 
                 d_ffn: int, d_model: int,
                 nhead: int, nlayers: int, config: FlowConfig = None
                 ):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        self.state_size = config.state_size
        self.num_proposals = config.num_proposals

        self.traj_encoding = nn.Linear(self.state_size, config.tf_d_model)
        self.traj_pos_embed = nn.Parameter(torch.randn(1, num_poses, config.tf_d_model) * 0.02)

        scorer_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ffn, 
            dropout=0.0, batch_first=True
        )
        self.scorer_attn = nn.TransformerDecoder(scorer_layer, num_layers=nlayers)

        def build_head(d_in, d_hidden):
            return nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 1)
            )

        self.heads = nn.ModuleDict({
            # --- Multipliers (安全因子) ---
            'no_at_fault_collisions': build_head(d_model, d_ffn),      # NC
            'drivable_area_compliance': build_head(d_model, d_ffn),    # DAC
            'driving_direction_compliance': build_head(d_model, d_ffn),# DDC
            'traffic_light_compliance': build_head(d_model, d_ffn),    # TLC
            
            # --- Weighted Scores (质量因子) ---
            'time_to_collision_within_bound': build_head(d_model, d_ffn),           # TTC
            'ego_progress': build_head(d_model, d_ffn),                # EP
            'lane_keeping': build_head(d_model, d_ffn),                # LK
            'history_comfort': build_head(d_model, d_ffn),             # HC
            'extended_comfort': build_head(d_model, d_ffn),                 # EC
        })

        # 权重配置
        self.weights = {
            'time_to_collision_within_bound': 5.0,
            'ego_progress': 5.0,
            'lane_keeping': 2.0,
            'history_comfort': 2.0,
            'extended_comfort': 2.0
        }
        self.sum_weights = sum(self.weights.values())

        self.multiplier_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 
                                'driving_direction_compliance', 'traffic_light_compliance']
        self.weighted_keys = ['time_to_collision_within_bound', 'ego_progress', 'lane_keeping', 
                              'history_comfort', 'extended_comfort']

    def forward(self, trajectories: torch.Tensor, 
                keyval: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectories: (B, K, L, 3)
            kv: (B, N, D)
        Returns:
            Dictionary containing best trajectory, scores, and detailed head outputs.
        """
        B, K, L, _ = trajectories.shape
        device = trajectories.device

        traj_token = self.traj_encoding(trajectories)
        traj_token_flat = traj_token.view(B * K, L, -1)
        traj_token_flat = traj_token_flat + self.traj_pos_embed

        # 交叉注意力
        tr_out = self.scorer_attn(traj_token_flat, keyval) # (B*K, L, D)

        # 3. 获取所有头的 Logits (B*K)
        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(tr_out).squeeze(-1)

        # --- 4. Log-Domain 计算总分 ---
        
        # A. Multiplier 部分: Sum of Log-Probs
        log_multiplier_sum = torch.zeros(B * K, device=device)
        for key in self.multiplier_keys:
            x = logits[key]
            # ln(sigmoid(x)) = -softplus(-x)
            log_p = -F.softplus(-x)
            log_multiplier_sum += log_p.sum(dim=-1)
        
        # B. Weighted 部分: Log(Sum of Weighted Probs)
        weighted_sum_prob = torch.zeros(B * K, device=device)
        for key in self.weighted_keys:
            x = logits[key]
            p = torch.sigmoid(x)
            weighted_sum_prob += self.weights[key] * p.sum(dim=-1)
        
        weighted_sum_prob /= self.sum_weights
        epsilon = 1e-8
        log_weighted_sum = torch.log(weighted_sum_prob + epsilon)

        # C. 最终 Log-Score
        log_epdms_score = log_multiplier_sum + log_weighted_sum # (B*K)
        log_epdms_score = log_epdms_score.view(B, K) # (B, K)

        # --- 5. 【关键修复】组装 pred_mult 和 pred_weighted ---
        
        # 我们需要输出两种形式以适配 Loss 和 分析：
        # 1. Logits (用于 BCEWithLogits Loss)
        # 2. Probs 或 Log-Probs (用于 MSE Loss 或 分析)
        
        # A. 组装 Multiplier (4项)
        mult_logits_list = [logits[k] for k in self.multiplier_keys]
        mult_probs_list = [torch.sigmoid(logits[k]) for k in self.multiplier_keys]
        
        # Stack -> (B*K, 4) -> View -> (B, K, 4)
        pred_mult_logits = torch.stack(mult_logits_list, dim=-1).view(B, K, -1)
        pred_mult_probs = torch.stack(mult_probs_list, dim=-1).view(B, K, -1)

        # B. 组装 Weighted (5项)
        weighted_logits_list = [logits[k] for k in self.weighted_keys]
        weighted_probs_list = [torch.sigmoid(logits[k]) for k in self.weighted_keys]
        
        pred_weighted_logits = torch.stack(weighted_logits_list, dim=-1).view(B, K, -1)
        pred_weighted_probs = torch.stack(weighted_probs_list, dim=-1).view(B, K, -1)

        # 6. 选择最佳轨迹
        best_idx = torch.argmax(log_epdms_score, dim=1, keepdim=True) # (B, 1)
        idx_expanded = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_poses, self.state_size)
        best_trajectory = torch.gather(trajectories, 1, idx_expanded).squeeze(1)
        
        # 7. 构建返回字典
        result = {
            'trajectory': best_trajectory,                  # (B, L, 3)
            'all_trajectories': trajectories,               # (B, K, L, 3)
            'log_epdms_score': log_epdms_score,             # (B, K) 总分
            
            # --- 核心输出：为了适配 Loss 函数 ---
            'pred_mult_logits': pred_mult_logits,           # (B, K, 4) 用于 BCE Loss
            'pred_mult_probs': pred_mult_probs,             # (B, K, 4) 用于分析
            
            'pred_weighted_logits': pred_weighted_logits,   # (B, K, 5) (MSE通常不需要logit，但保留以防万一)
            'pred_weighted_probs': pred_weighted_probs,     # (B, K, 5) 用于 MSE Loss
            
            'best_idx': best_idx                            # (B, 1)
        }
        
        return result

class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self._config = config
        self.num_proposals = config.num_proposals
        self.image_backbone = ImgEncoder(config)

        self.scene_embeds = nn.Parameter(torch.randn(1, self._config.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6)
        
        self.hist_encoding = nn.Linear(config.hist_ego_dim, config.tf_d_model)
        self.hist_pos_embed = nn.Parameter(torch.randn(1, config.hist_ego_len, config.tf_d_model) * 0.02)

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

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # image, bev, agent, traj, score
        camera_feature: torch.Tensor = features["image"]
        status_feature: torch.Tensor = features["ego_status"]
        batch_size = status_feature.shape[0]

        scene_token = self.scene_embeds.repeat(batch_size, 1, 1, 1)

        img_token = self.image_backbone(camera_feature, scene_token)
        # img_token = torch.randn(batch_size, 
        #                         self._config.num_cams * self._config.num_scene_tokens, 
        #                         self._config.tf_d_model)
        
        ego_token = self.hist_encoding(status_feature)
        ego_token = ego_token + self.hist_pos_embed

        keyval = torch.cat([img_token, ego_token], dim=1)
        kv_expanded = keyval.repeat_interleave(self.num_proposals, dim=0)

        trajectory = self._trajectory_head(kv_expanded)

        output = self._score_head(trajectory, kv_expanded)
        output['env_kv'] = kv_expanded

        return output

