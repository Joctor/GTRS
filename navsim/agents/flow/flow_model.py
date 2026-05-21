import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.flow.flow_config import FlowConfig
from navsim.agents.flow.dit import MFDiT

from .layers.image_encoder.dinov2_lora import ImgEncoder
from navsim.agents.flow.vit import TransformerDecoder, TransformerDecoderScorer

class FlowHead(nn.Module):
    def __init__(self, num_poses: int, 
                 d_ffn: int, d_model: int,
                 nhead: int, nlayers: int, config: FlowConfig = None,
                 scorehead=None
                 ):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        self.state_size = config.state_size
        self.num_proposals = config.num_proposals

        self.mfdit = MFDiT(input_size=num_poses * config.state_size, 
              num_poses=config.num_proposals, 
              hidden_size=d_model, 
              depth=nlayers, 
              num_heads=nhead)

    def forward(self, img_token, ego_token):
        B = img_token.shape[0]
        device = img_token.device
        K = self.num_proposals

        x_t = torch.randn(B, K, self.num_poses * self.state_size, device=device) 
        
        num_steps = 10 
        dt = 1.0 / num_steps

        for i in range(num_steps):
            # 构造时间步 t
            # 形状必须是 (B, 1, 1)，对应合并后的 Batch 维度
            t_val = torch.full((B, 1, 1), i * dt, device=device)
            
            target_score = torch.ones((B, 9), device=device)
            v_t_cond = self.mfdit(x_t, t_val, img_token, ego_token, target_score)

            zero_score = torch.zeros((B, 9), device=device)
            v_t_uncond = self.mfdit(x_t, t_val, img_token, ego_token, zero_score)

            v_t = v_t_uncond + 2 * (v_t_cond - v_t_uncond)
            
            # 更新轨迹
            x_t = x_t + v_t * dt

        trajectories = x_t.view(B, K, self.num_poses, -1)
        
        return trajectories
    
    def get_flow_loss(self, targets, predictions, good_mask=None, tensor_df=None):
        if good_mask is None:
            img_token = predictions['img_token']
            ego_token = predictions['ego_token']
            device = img_token.device

            gt_trajectory = targets['trajectory'].float()
            gt_score = targets['gt_score'].float().nan_to_num(nan=0.0)
        else:
            img_token = predictions['img_token'][good_mask]
            ego_token = predictions['ego_token'][good_mask]
            device = img_token.device

            gt_trajectory = predictions['trajectory'][good_mask]

            gt_score = tensor_df[good_mask]
            zero_col = torch.zeros(gt_score.shape[0], 1, device=device, dtype=gt_score.dtype)
            gt_score = torch.cat([gt_score[:, :-1], zero_col, gt_score[:, -1:]], dim=1)
        
        B = gt_trajectory.shape[0]
        
        drop_mask = torch.rand(B, 1) < 0.1 
        train_pdm_score = torch.where(drop_mask.to(device), torch.zeros_like(gt_score, device=device), gt_score)
        x_1 = gt_trajectory.view(B, 1, -1)
        x_0 = torch.randn(B, self.num_proposals, self.num_poses * self.state_size, device=device) 
        
        t = torch.rand(x_0.shape[0], 1, 1, device=device)
        # 插值
        z_t = x_0 + t * (x_1 - x_0)
        # 加噪声 (可选，增加鲁棒性)
        # z_t = z_t + 0.02 * torch.randn_like(z_t)
        
        # 目标速度
        cvf = (x_1 - x_0)
        
        # 预测
        v_t = self.mfdit(z_t, t, img_token, ego_token, train_pdm_score)
        
        flow_loss = F.mse_loss(v_t, cvf)

        return flow_loss
    
class TrajHead(nn.Module):
    def __init__(self, num_poses: int, 
                 d_ffn: int, d_model: int,
                 nhead: int, nlayers: int, config: FlowConfig = None
                 ):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        self.state_size = config.state_size
        self.num_proposals = config.num_proposals

        self.traj_encoding = nn.Sequential(
            nn.Linear(num_poses * config.state_size, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        
        self.traj_decoder = TransformerDecoder(config)

        self.output_layer = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.LayerNorm(d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.LayerNorm(d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, self.num_poses * config.state_size),
            )
        
    def forward(self, trajectories: torch.Tensor, 
            img_token) -> Dict[str, torch.Tensor]:
        
        B, K, L, _ = trajectories.shape
        device = trajectories.device

        trajectories = trajectories.view(B, K, -1)
        traj_token = self.traj_encoding(trajectories)
        
        tr_out = self.traj_decoder(traj_token, img_token)
        proposal = self.output_layer(tr_out).view(B, K, L, -1)

        return proposal

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

        self.traj_encoding = nn.Sequential(
            nn.Linear(num_poses * config.state_size, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

        self.scorer_attn = TransformerDecoderScorer(config)

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
            'lane_keeping': build_head(d_model, d_ffn),                # LK
            'history_comfort': build_head(d_model, d_ffn),             # HC
            'two_frame_extended_comfort': build_head(d_model, d_ffn),                 # EC
            'ego_progress': build_head(d_model, d_ffn),                # EP
        })

        # 权重配置
        self.weights = {
            'time_to_collision_within_bound': 5.0,
            'ego_progress': 5.0,
            'lane_keeping': 2.0,
            'history_comfort': 2.0,
            'two_frame_extended_comfort': 2.0
        }
        self.sum_weights = sum(self.weights.values())

        self.multiplier_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 
                                'driving_direction_compliance', 'traffic_light_compliance']
        self.weighted_keys = ['time_to_collision_within_bound', 'lane_keeping', 
                              'history_comfort', 'two_frame_extended_comfort', 'ego_progress']

    def forward(self, trajectories: torch.Tensor, 
                img_token, ego_token) -> Dict[str, torch.Tensor]:
        
        B, K, L, _ = trajectories.shape
        device = trajectories.device

        trajectories = trajectories.detach().view(B, K, -1)
        traj_token = self.traj_encoding(trajectories)

        tr_out = self.scorer_attn(traj_token, img_token) # (B, K, D)
        tr_out = tr_out + ego_token[:, -1, :].unsqueeze(1)
        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(tr_out).squeeze(-1)

        # --- 4. Log-Domain 计算总分 ---
        
        # A. Multiplier 部分: Sum of Log-Probs
        log_multiplier_sum = torch.zeros(B, K, device=device)
        for key in self.multiplier_keys:
            x = logits[key]
            # ln(sigmoid(x)) = -softplus(-x)
            log_p = -F.softplus(-x)
            log_multiplier_sum += log_p
        
        # B. Weighted 部分: Log(Sum of Weighted Probs)
        weighted_sum_prob = torch.zeros(B, K, device=device)
        for key in self.weighted_keys:
            x = logits[key]
            p = torch.sigmoid(x)
            weighted_sum_prob += self.weights[key] * p
        
        weighted_sum_prob /= self.sum_weights
        epsilon = 1e-8
        log_weighted_sum = torch.log(weighted_sum_prob.clamp(min=epsilon))

        # C. 最终 Log-Score
        log_epdms_score = log_multiplier_sum + log_weighted_sum # (B, K)

        # 1. 获取最高分数和对应的索引
        # max_scores: (B,), 包含每个 batch 中的最高分
        # best_indices: (B,), 包含每个 batch 中最高分轨迹的索引
        max_scores, best_indices = torch.max(log_epdms_score, dim=1)

        # 2. 准备用于高级索引的 batch 索引
        # 创建一个形如 [0, 1, 2, ..., B-1] 的张量
        batch_indices = torch.arange(B, device=device)

        # 3. 提取最佳轨迹和所有对应的 logit 分数
        # 使用 best_indices 从 K 个候选中选出每个 batch 最佳的那个
        best_trajectories = trajectories.view(B, K, L, -1)[batch_indices, best_indices] # (B, L, 3)

        pred_logits = torch.stack(
            [logits[name][batch_indices, best_indices] for name in logits], dim=1)
        
        # 7. 构建返回字典
        result = {
            'trajectory': best_trajectories,                  # (B, L, 3)
            'max_scores': max_scores,             # (B, 1) 总分
            'all_scores':log_epdms_score,                # (B, K)
            'pred_logits': pred_logits,                     # (B, 9)
            'best_idx': best_indices                            # (B, 1)
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

        self._traj_head = TrajHead(
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

        self._flow_head = FlowHead(
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

        scene_token = self.scene_embeds.repeat(batch_size, 1, 1, 1)

        img_token = self.image_backbone(camera_feature, scene_token)
        # img_token = torch.randn(batch_size, 
        #                         self._config.num_cams * self._config.num_scene_tokens, 
        #                         self._config.tf_d_model)
        
        ego_token = self.hist_encoding(status_feature)
        ego_token = ego_token + self.hist_pos_embed

        flow_proposal = self._flow_head(img_token, ego_token)

        traj_proposal = self._traj_head(flow_proposal, img_token)

        output = self._score_head(traj_proposal, img_token, ego_token)

        output['img_token'] = img_token
        output['ego_token'] = ego_token
        
        output['flow_proposal'] = flow_proposal
        output['traj_proposal'] = traj_proposal

        return output

