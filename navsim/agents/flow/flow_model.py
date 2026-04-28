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

    @torch.no_grad()
    def forward(self, img_token, ego_token):
        """
        生成多候选轨迹
        Args:
            constant_velocity: (B, num_poses, 3)
            img_token: (B, L_img, C)
            ego_token: (B, L_ego, C)
        Returns:
            trajectories: (B, num_candidates, num_poses, 3)
        """
        B = img_token.shape[0]
        device = img_token.device
        K = self.num_proposals
        
        # ==========================================
        # 1. 准备数据并扩展维度
        # ==========================================
        
        # 1.1 扩展 constant_velocity: (B, num_poses, 3) -> (B, K, num_poses, 3)
        x_t = torch.randn(B, K, self.num_poses, self.state_size, device=device) 
        
        # 1.2 扩展条件 Token: (B, L, C) -> (B, K, L, C)
        img_token_expanded = img_token.unsqueeze(1).repeat(1, K, 1, 1)
        ego_token_expanded = ego_token.unsqueeze(1).repeat(1, K, 1, 1)

        # ==========================================
        # 2. 维度重塑 (Reshape): (B, K, ...) -> (B*K, ...)
        # ==========================================
        # 将 Batch 维度和 Candidate 维度合并
        x_t = x_t.view(B * K, self.num_poses, -1)             # (B*K, 8, 3)
        img_token_flat = img_token_expanded.view(B * K, -1, img_token_expanded.shape[-1]) # (B*K, L_img, C)
        ego_token_flat = ego_token_expanded.view(B * K, -1, ego_token_expanded.shape[-1]) # (B*K, L_ego, C)

        # ==========================================
        # 3. 迭代去噪 (Rectified Flow 推理)
        # ==========================================
        num_steps = 10 
        dt = 1.0 / num_steps

        for i in range(num_steps):
            # 构造时间步 t
            # 形状必须是 (B*K, 1, 1)，对应合并后的 Batch 维度
            t_val = torch.full((B * K, 1, 1), i * dt, device=device)
            
            # 预测速度场
            # 输入形状: (B*K, 8, 3)
            # 输出形状: (B*K, 8, 3)
            target_score = torch.ones((B * K, 9), device=device)
            v_t_cond = self.mfdit(x_t, t_val, img_token_flat, ego_token_flat, target_score)

            zero_score = torch.zeros((B * K, 9), device=device)
            v_t_uncond = self.mfdit(x_t, t_val, img_token_flat, ego_token_flat, zero_score)

            v_t = v_t_uncond + 2 * (v_t_cond - v_t_uncond)
            
            # 更新轨迹
            x_t = x_t + v_t * dt

        # ==========================================
        # 4. 恢复维度 (Reshape Back): (B*K, ...) -> (B, K, ...)
        # ==========================================
        # 将结果还原为 (B, K, num_poses, 3) 以便返回
        trajectories = x_t.view(B, K, self.num_poses, -1)
        
        return trajectories
    
    def get_flow_loss(self, targets, predictions):
        # kv: (batch_size, num_bev_tokens + 4, config.tf_d_model)
        # gt_trajectory: (batch_size, 8, 3)
        img_token = predictions['img_token']
        ego_token = predictions['ego_token']
        flow_proposal = predictions['flow_proposal']
        gt_trajectory = targets['trajectory'].float()
        B = img_token.shape[0]
        K = flow_proposal.shape[1]
        device = img_token.device

        gt_score = targets['gt_score'].float()
        drop_mask = torch.rand(B, 1) < 0.1 
        train_pdm_score = torch.where(drop_mask, torch.zeros_like(gt_score), gt_score)

        x_1 = gt_trajectory
        x_0 = torch.randn(B, self.num_poses, self.state_size, device=device) 
        
        t = torch.rand(x_0.shape[0], 1, 1, device=device)
        # 插值
        z_t = x_0 + t * (x_1 - x_0)
        # 加噪声 (可选，增加鲁棒性)
        z_t = z_t + 0.02 * torch.randn_like(z_t)
        
        # 目标速度
        cvf = (x_1 - x_0)
        
        # 预测
        v_t = self.mfdit(z_t, t, img_token, ego_token, train_pdm_score)
        
        flow_loss = F.mse_loss(v_t, cvf)

        dist_matrix = torch.norm(flow_proposal.unsqueeze(2) - flow_proposal.unsqueeze(1), dim=-1).sum(dim=-1)
        mask = torch.triu(torch.ones(K, K, device=device), diagonal=1)
        pairwise_penalty = torch.exp(-dist_matrix)
        diversity_loss = (pairwise_penalty * mask).sum() / (0.5 * K * (K - 1))

        gt_expanded = gt_trajectory.unsqueeze(1)
        dists = torch.norm(flow_proposal - gt_expanded, dim=-1).sum(dim=-1)
        mindist_loss = torch.min(dists, dim=1)[0].mean()

        return flow_loss, diversity_loss, mindist_loss
    
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

        self.proposal_encoding = nn.Sequential(
            nn.Linear(config.trajectory_sampling.num_poses * config.state_size, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        
        self.traj_decoder = TransformerDecoder(config)

        self.output_layer = nn.ModuleList()
        for i in range(config.ref_num):
            self.output_layer.append(
                nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.LayerNorm(d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_ffn),
                nn.LayerNorm(d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, config.trajectory_sampling.num_poses * config.state_size),
            )
            )
        
    def forward(self, proposal: torch.Tensor, 
            img_token) -> Dict[str, torch.Tensor]:

        B, K, L, _ = proposal.shape
        device = proposal.device

        proposal = proposal.view(B, K, -1)
        #(B,K,256)
        proposal_token = self.proposal_encoding(proposal)
        
        trout_list = self.traj_decoder(proposal_token, img_token)

        proposal_list = []
        for i in range(self.config.ref_num):
            trout = trout_list[i]
            proposal = self.output_layer[i](trout)
            proposal = proposal.reshape(B, K, L, -1)
            proposal_list.append(proposal)

        return proposal_list
    
    def get_traj_loss(self, targets, predictions):
        traj_proposal = predictions['traj_proposal']
        gt_trajectory = targets['trajectory'].float()
        B = gt_trajectory.shape[0]
        K = traj_proposal[0].shape[1]
        device = gt_trajectory.device

        diversity_loss = 0
        mindist_loss = 0
        for traj_i in traj_proposal:
            dist_matrix = torch.norm(traj_i.unsqueeze(2) - traj_i.unsqueeze(1), dim=-1).sum(dim=-1)
            mask = torch.triu(torch.ones(K, K, device=device), diagonal=1)
            pairwise_penalty = torch.exp(-dist_matrix)
            diversity_loss += (pairwise_penalty * mask).sum() / (0.5 * K * (K - 1))

            gt_expanded = gt_trajectory.unsqueeze(1)
            dists = torch.norm(traj_i - gt_expanded, dim=-1).sum(dim=-1)
            mindist_loss += torch.min(dists, dim=1)[0].mean()

        return diversity_loss, mindist_loss

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
                nn.Linear(num_poses * config.state_size, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
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
        """
        Args:
            trajectories: (B, K, L, 3)
        Returns:
            Dictionary containing best trajectory, scores, and detailed head outputs.
        """
        # GT轨迹进来的变换
        if len(trajectories.shape) == 3:
            trajectories = trajectories.unsqueeze(1)  # (B, 1, L, 3)
        
        B, K, L, _ = trajectories.shape
        device = trajectories.device

        trajectories = trajectories.view(B, K, -1)
        traj_token = self.traj_encoding(trajectories)
        
        # (B, 64+4, D)
        keyval = img_token + ego_token[:,-1,:].unsqueeze(1)

        # 交叉注意力
        tr_out = self.scorer_attn(traj_token, keyval) # (B, K, D)

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
        best_trajectories = trajectories.view(B, K, L, 3)[batch_indices, best_indices] # (B, L, 3)

        pred_logits = torch.stack(
            [logits[name][batch_indices, best_indices] for name in logits], dim=1)
        
        # 7. 构建返回字典
        result = {
            'trajectory': best_trajectories,                  # (B, L, 3)
            'log_epdms_score': max_scores,             # (B, 1) 总分
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
        
        self._flow_head = FlowHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            config=config
        )

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

        #(B,K,8,3)
        flow_proposal = self._flow_head(img_token, ego_token)
        #4*(B,K,8,3)
        traj_proposal = self._traj_head(flow_proposal, img_token)
        #(B,5*K,8,3)
        all_proposal = torch.cat([flow_proposal, torch.cat(traj_proposal, dim=1)], dim=1)

        output = self._score_head(all_proposal, img_token, ego_token)

        output['img_token'] = img_token
        output['ego_token'] = ego_token
        
        output['flow_proposal'] = flow_proposal
        output['traj_proposal'] = traj_proposal

        return output

