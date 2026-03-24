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

    def forward(self, trajectories: torch.Tensor, trajectory_feature: torch.Tensor, 
                kv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectories: (B, K, L, 3)
            trajectory_feature: (B*K, L, D) 来自 FlowHead
            kv: (B, N, D)
        Returns:
            Dictionary containing best trajectory, scores, and detailed head outputs.
        """
        B, K, L, _ = trajectories.shape
        device = trajectories.device
        
        # 1. 扩展 KV
        kv_expanded = kv.repeat_interleave(K, dim=0) # (B*K, N, D)
        
        # 2. 二次环境查询 (Refine) & 池化
        refined_features = self.scorer_refine_attn(tgt=trajectory_feature, memory=kv_expanded)
        pooled_feat = refined_features.mean(dim=1) # (B*K, D)

        # 3. 获取所有头的 Logits (B*K)
        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(pooled_feat).squeeze(-1)

        # --- 4. Log-Domain 计算总分 ---
        
        # A. Multiplier 部分: Sum of Log-Probs
        log_multiplier_sum = torch.zeros(B * K, device=device)
        for key in self.multiplier_keys:
            x = logits[key]
            # ln(sigmoid(x)) = -softplus(-x)
            log_p = -F.softplus(-x)
            log_multiplier_sum += log_p
        
        # B. Weighted 部分: Log(Sum of Weighted Probs)
        weighted_sum_prob = torch.zeros(B * K, device=device)
        for key in self.weighted_keys:
            x = logits[key]
            p = torch.sigmoid(x)
            weighted_sum_prob += self.weights[key] * p
        
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
        idx_expanded = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, 3)
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

