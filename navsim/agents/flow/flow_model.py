import numpy as np
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

    def forward(self, img_token, ego_token, weighted_vocab_token):
        B = img_token.shape[0]
        device = img_token.device
        K = self.num_proposals

        x_t = torch.randn(B*K, self.num_poses, self.state_size, device=device)
        
        img_token_flat = img_token.repeat_interleave(K, dim=0)
        ego_token_flat = ego_token.repeat_interleave(K, dim=0)
        weighted_vocab_token_flat = weighted_vocab_token.repeat_interleave(K, dim=0)
        
        num_steps = 10 
        dt = 1.0 / num_steps

        for i in range(num_steps):
            # 构造时间步 t
            # 形状必须是 (B, 1, 1)，对应合并后的 Batch 维度
            t_val = torch.full((B*K, 1, 1), i * dt, device=device)
            
            v_t_cond = self.mfdit(x_t, t_val, img_token_flat, ego_token_flat, weighted_vocab_token_flat)

            zeros_token = torch.zeros_like(weighted_vocab_token_flat, device=device)
            v_t_uncond = self.mfdit(x_t, t_val, img_token_flat, ego_token_flat, zeros_token)

            v_t = v_t_uncond + 2 * (v_t_cond - v_t_uncond)
            
            # 更新轨迹
            x_t = x_t + v_t * dt

        trajectories = x_t.view(B, K, self.num_poses, -1)
        
        return trajectories
    
    def get_flow_loss(self, predictions, gt_traj, vocab_score):
        B = gt_traj.shape[0]
            img_token = predictions['img_token']
            ego_token = predictions['ego_token']
        vocab_token = predictions['vocab_token']
            device = img_token.device

        multiplier_logits = vocab_score[..., :4]
        log_multiplier_sum = -F.softplus(-multiplier_logits).sum(dim=-1) 

        weighted_logits = vocab_score[..., 4:]
        weighted_probs = torch.sigmoid(weighted_logits)
        # ttc,lk,hc,ep
        weighted_sum_prob = (weighted_probs * torch.tensor([5.0, 2.0, 2.0, 5.0], device=device)).sum(dim=-1) / 14.0
        log_weighted_sum = torch.log(weighted_sum_prob.clamp(min=1e-8))
        
        combined_score = log_multiplier_sum + log_weighted_sum
        #(B, K, 1)
        score_weights = combined_score.softmax(dim=-1).unsqueeze(-1)
        #(B, K, D)
        weighted_vocab_token = vocab_token * score_weights
        
        drop_mask = torch.rand(B, 1, 1) < 0.2
        weighted_vocab_token = torch.where(drop_mask.to(device), 
                                           torch.zeros_like(weighted_vocab_token, device=device), 
                                           weighted_vocab_token)
        
        x_1 = gt_traj

        x_0 = torch.randn(B, self.num_poses, self.state_size, device=device)
        
        t = torch.rand(x_0.shape[0], 1, 1, device=device)
        # 插值
        z_t = x_0 + t * (x_1 - x_0)
        # 加噪声 (可选，增加鲁棒性)
        # z_t = z_t + 0.02 * torch.randn_like(z_t)
        
        # 目标速度
        cvf = (x_1 - x_0)
        
        # 预测
        v_t = self.mfdit(z_t, t, img_token, ego_token, weighted_vocab_token)
        
        flow_loss = F.mse_loss(v_t, cvf)

        return flow_loss
    
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
            # 'two_frame_extended_comfort': build_head(d_model, d_ffn),                 # EC
            'ego_progress': build_head(d_model, d_ffn),                # EP
        })

        # 权重配置
        self.weights = {
            'time_to_collision_within_bound': 5.0,
            'ego_progress': 5.0,
            'lane_keeping': 2.0,
            'history_comfort': 2.0,
            # 'two_frame_extended_comfort': 2.0
        }
        self.sum_weights = sum(self.weights.values())

        self.multiplier_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 
                                'driving_direction_compliance', 'traffic_light_compliance']
        self.weighted_keys = ['time_to_collision_within_bound', 'lane_keeping', 
                              'history_comfort', 'ego_progress']

    def forward(self, traj_token: torch.Tensor, 
                img_token, ego_token) -> Dict[str, torch.Tensor]:
        
        B, K, _ = traj_token.shape
        device = traj_token.device

        tr_out = self.traj_decoder(traj_token, img_token)

        scorer_out = self.scorer_attn(tr_out, img_token) # (B, K, D)

        proposal = self.output_layer(scorer_out).view(B, K, self.num_poses, -1)

        scorer_out = scorer_out + ego_token
        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(scorer_out).squeeze(-1)

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
        best_proposal = proposal[batch_indices, best_indices] # (B, L, 4)

        # 4. 提取所有候选轨迹的 logits (B, K, 8)
        all_pred_logits = torch.stack([logits[name] for name in self.heads.keys()], dim=-1)

        # 5. （可选）如果你依然需要保留原本只含最佳轨迹分数的 pred_logits (B, 8)
        best_pred_logits = all_pred_logits[batch_indices, best_indices] 
        
        # 7. 构建返回字典
        result = {
            'all_proposals':proposal,
            'best_proposal': best_proposal,              # (B, L, 4
            'max_scores': max_scores,             # (B, 1) 总分
            'all_scores':log_epdms_score,                # (B, K)
            'all_pred_logits': all_pred_logits,  
            'pred_logits': best_pred_logits,                     # (B, 8)
            'best_indices': best_indices                            # (B, 1)
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

        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(config.vocab_path)),
            requires_grad=False
        )

        self.traj_encoding = nn.Sequential(
            nn.Linear(config.trajectory_sampling.num_poses * config.state_size, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
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

    def cumsum_traj(self, model_output):
        """
        model_output: 模型预测输出的 Tensor，形状为 (B, 8, 4)
                    [..., 0]=x特征, [..., 1]=y特征, [..., 2]=sin_theta, [..., 3]=cos_theta
        """
        # ==============================
        # 第一步：提取 X, Y 和 Sin/Cos 特征通道
        # ==============================
        x_final = model_output[..., 0]
        y_final = model_output[..., 1]
        sin_theta = model_output[..., 2]
        cos_theta = model_output[..., 3]
        
        # ==============================
        # 第二步：还原 X 和 Y 的物理坐标（保持原有逻辑）
        # ==============================
        x_log = x_final * (self._config.x_std + 1e-8) + self._config.x_mean
        y_diff = y_final * (self._config.y_std + 1e-8) + self._config.y_mean

        x_diff = torch.sign(x_log) * torch.expm1(torch.abs(x_log))
        x_recovered = torch.cumsum(x_diff, dim=1)
        y_recovered = torch.cumsum(y_diff, dim=1)
        
        # ==============================
        # 第三步：还原绝对角度 Theta
        # ==============================
        # 1. 利用 atan2(sin, cos) 将 sin/cos 映射回 [-pi, pi] 范围内的角度差值
        delta_theta = torch.atan2(sin_theta, cos_theta)
        
        # 2. 沿着时间步累加角度差值，得到绝对的航向角
        theta_recovered = torch.cumsum(delta_theta, dim=1)
        
        return torch.stack([x_recovered, y_recovered, theta_recovered], dim=-1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # image, bev, agent, traj, score
        camera_feature: torch.Tensor = features["image"]
        status_feature: torch.Tensor = features["ego_status"][:, -1:]
        
        batch_size = status_feature.shape[0]

        scene_token = self.scene_embeds.repeat(batch_size, 1, 1, 1)

        img_token = self.image_backbone(camera_feature, scene_token)
        # img_token = torch.randn(batch_size, 
        #                         self._config.num_cams * self._config.num_scene_tokens, 
        #                         self._config.tf_d_model)
        
        ego_token = self.hist_encoding(status_feature)
        #(B,21476,8,4)
        vocab_proposal = self.vocab.unsqueeze(0).expand(batch_size, -1, -1, -1)
        #(B,21476,D)
        vocab_token = self.traj_encoding(vocab_proposal.view(batch_size, -1, self._config.trajectory_sampling.num_poses * self._config.state_size))

        output = {}

        output['vocab'] = self._score_head(vocab_token, img_token, ego_token)
        output['vocab']['best_vocab'] = vocab_proposal[torch.arange(batch_size), output['vocab']['best_indices']]

        score_weights = output['vocab']['all_scores'].softmax(dim=-1).unsqueeze(-1)
        #(B,21476,D)
        weighted_vocab_token = vocab_token * score_weights

        flow_proposal = self._flow_head(img_token, ego_token, weighted_vocab_token)

        flow_token = self.traj_encoding(flow_proposal.view(batch_size, self.num_proposals, -1))

        output['flow'] = self._score_head(flow_token, img_token, ego_token)

        output['img_token'] = img_token
        output['ego_token'] = ego_token
        output['vocab_token'] = vocab_token

        which_better = output['vocab']['max_scores'] > output['flow']['max_scores']
        which_better = which_better.view(-1, 1, 1) 
        output['max_scores'] = torch.where(which_better, output['vocab']['max_scores'], output['flow']['max_scores'])
        
        output['trajectory'] = torch.where(which_better, output['vocab']['best_vocab'], output['flow']['best_proposal'])
        output['trajectory'] = self.cumsum_traj(output['trajectory'])

        return output

