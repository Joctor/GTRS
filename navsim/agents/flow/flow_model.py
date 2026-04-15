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
        
        self.twinflow_lambda = 0.5 

    @torch.no_grad()
    def forward(self, constant_velocity, img_token, ego_token):
        """
        生成多候选轨迹
        Args:
            constant_velocity: (B, num_poses, 3)
            img_token: (B, L_img, C)
            ego_token: (B, L_ego, C)
        Returns:
            trajectories: (B, num_candidates, num_poses, 3)
        """
        B = constant_velocity.shape[0]
        device = constant_velocity.device
        K = self.num_proposals
        
        # ==========================================
        # 1. 准备数据并扩展维度
        # ==========================================
        
        # 1.1 扩展 constant_velocity: (B, num_poses, 3) -> (B, K, num_poses, 3)
        x_t = constant_velocity.unsqueeze(1).repeat(1, K, 1, 1)
        
        # 1.2 扩展条件 Token: (B, L, C) -> (B, K, L, C)
        img_token_expanded = img_token.unsqueeze(1).repeat(1, K, 1, 1)
        ego_token_expanded = ego_token.unsqueeze(1).repeat(1, K, 1, 1)

        # 1.3 生成独立噪声
        pos_noise_sigma = 1.0
        yaw_noise_sigma = 0.1
        noise_pos = torch.randn(B, K, self.num_poses, 2, device=device) * pos_noise_sigma
        noise_yaw = torch.randn(B, K, self.num_poses, 1, device=device) * yaw_noise_sigma
        noise = torch.cat([noise_pos, noise_yaw], dim=-1)
        
        # 1.4 初始化 x_t
        x_t = x_t + noise

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
            v_t = self.mfdit(x_t, t_val, img_token_flat, ego_token_flat)
            
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
        constant_velocity = predictions['constant_velocity']
        gt_trajectory = targets['trajectory'].float()

        B = img_token.shape[0]
        device = img_token.device

        split_idx = int(B * self.twinflow_lambda)
        
        # 防止 split_idx 为 0 或 B 导致某一部分没数据
        if split_idx == 0: split_idx = 1
        if split_idx == B: split_idx = B - 1

        # 数据切片
        x_real_twin = gt_trajectory[:split_idx]
        x_const_twin = constant_velocity[:split_idx]
        img_twin = img_token[:split_idx]
        ego_twin = ego_token[:split_idx]
        
        x_real_base = gt_trajectory[split_idx:]
        x_const_base = constant_velocity[split_idx:]
        img_base = img_token[split_idx:]
        ego_base = ego_token[split_idx:]

        flow_loss = 0.0

        if x_real_base.shape[0] > 0:
            x_1 = x_real_base
            x_0 = x_const_base
            
            t = torch.rand(x_0.shape[0], 1, 1, device=device)
            # 插值
            z_t = x_0 + t * (x_1 - x_0)
            # 加噪声 (可选，增加鲁棒性)
            z_t = z_t + 0.02 * torch.randn_like(z_t)
            
            # 目标速度
            cvf = (x_1 - x_0)
            
            # 预测
            v_t = self.mfdit(z_t, t, img_base, ego_base)
            
            flow_loss += F.mse_loss(v_t, cvf)
        
        if x_real_twin.shape[0] > 0:
            x_1 = x_real_twin
            x_0 = x_const_twin
            
            # --- A. 生成假数据 (Fake Data Generation) ---
            # 对应公式: x_fake = z - v_theta(z, 0)
            # 注意：这里我们用 constant_velocity 作为基准噪声/起点
            z_noise = x_0 # 在这里我们将 constant_velocity 视为噪声源 z
            
            # 预测 t=0 时的速度 (从噪声到数据的初始速度)
            # 时间输入为 0
            t_zero = torch.zeros(z_noise.shape[0], 1, 1, device=device)
            v_z = self.mfdit(z_noise, t_zero, img_twin, ego_twin)
            
            # 生成假目标: x_fake
            # 这里减去速度，是因为我们定义速度场是从 x_0 指向 x_1
            # 如果想反推 x_0 应该长什么样才能一步到 x_1，逻辑类似
            # 但根据 TwinFlow 论文，是生成一个 "Fake Target"
            x_fake = z_noise + v_z # 这里符号取决于你的速度场定义，通常是 x_0 + v * dt
            
            # --- B. 自对抗损失 (L_adv) ---
            # 学习从噪声到 x_fake 的路径 (负时间步)
            t_prime = torch.rand(x_0.shape[0], 1, 1, device=device) # t' ~ U(0,1)
            
            # 构造负向轨迹的扰动样本
            # x_t_fake = (1-t') * z_new + t' * x_fake  (这里简化插值逻辑)
            x_t_fake = z_noise + t_prime * (x_fake - z_noise) 
            x_t_fake = x_t_fake + 0.02 * torch.randn_like(x_t_fake) # 加噪
            
            # 目标速度: 从 z_noise 到 x_fake
            v_target_adv = x_fake - z_noise
            
            # 模型预测：输入负时间 -t'
            v_pred_adv = self.mfdit(x_t_fake, -t_prime, img_twin, ego_twin)
            
            flow_loss += F.mse_loss(v_pred_adv, v_target_adv)
            
            # --- C. 矫正损失 (L_rectify) ---
            # 匹配正负轨迹的速度
            # 1. 计算 v_fake (在负轨迹上)
            v_fake_val = self.mfdit(x_t_fake.detach(), -t_prime, img_twin, ego_twin)
            
            # 2. 计算 v_real (在正轨迹上，对应相同的时间点 t')
            # 注意：这里我们需要构造正向轨迹的对应点，或者复用 x_t_fake 的位置
            # 为了简化，我们假设在相同的位置比较速度场
            v_real_val = self.mfdit(x_t_fake.detach(), t_prime, img_twin, ego_twin)
            
            # 速度差 Delta v
            delta_v = v_real_val - v_fake_val
            
            # 矫正目标：让 v_z (初始预测) 去逼近 v_z + Delta_v
            # 这意味着模型要学习修正这个速度差
            target_rectify = (v_z + delta_v).detach()
            
            flow_loss += F.mse_loss(v_z, target_rectify)

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

        self.traj_encoding = nn.Sequential(
                nn.Linear(num_poses * config.state_size, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )

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
        keyval = torch.cat([img_token, ego_token], dim=1)

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
            'all_trajectories': trajectories,               # (B, K, L, 3)
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

    def compute_constant_velocity(self, status_feature: torch.Tensor) -> torch.Tensor:
        """
        输入: status_feature (B, 4, 11)
        输出: Trajectory (包含 (B, num_poses, 3) 的 poses)
        """
        device = status_feature.device
        B = status_feature.shape[0]
        
        dt = self._config.trajectory_sampling.interval_length
        num_poses = self._config.trajectory_sampling.num_poses
        
        # 1. 提取当前和上一帧数据
        status_curr = status_feature[:, -1, :]  # (B, 11)
        status_prev = status_feature[:, -2, :]  # (B, 11)

        # 2. 提取物理量 (确保在设备上)
        current_x = status_curr[:, 0]           # (B,)
        current_y = status_curr[:, 1]           # (B,)
        current_yaw = status_curr[:, 2]         # (B,)
        
        prev_yaw = status_prev[:, 2]            # (B,)

        current_vx = status_curr[:, 3]          # (B,)
        current_vy = status_curr[:, 4]          # (B,)

        # 3. 计算角速度 (Omega)
        delta_yaw = current_yaw - prev_yaw
        
        # 处理 -pi 到 pi 的跳变 (使用 torch 函数)
        # 公式: (x + pi) % (2*pi) - pi
        delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        
        angular_velocity = delta_yaw / dt       # (B,)

        # 4. 向量化生成轨迹 (去掉循环)
        # 创建时间序列向量: [1*dt, 2*dt, ..., num_poses*dt]
        # 形状: (num_poses,)
        time_steps = torch.arange(1, num_poses + 1, device=device) * dt
        
        # 利用广播机制计算
        # current_x 是 (B, 1), time_steps 是 (num_poses,) -> 结果 (B, num_poses)
        # 我们需要先 unsqueeze 变成 (B, 1) 才能和 (num_poses,) 广播
        future_x = current_x.unsqueeze(1) + current_vx.unsqueeze(1) * time_steps.unsqueeze(0)
        future_y = current_y.unsqueeze(1) + current_vy.unsqueeze(1) * time_steps.unsqueeze(0)
        future_yaw = current_yaw.unsqueeze(1) + angular_velocity.unsqueeze(1) * time_steps.unsqueeze(0)
        
        # 5. 堆叠成 (B, num_poses, 3)
        # stack 最后一个维度
        poses = torch.stack([future_x, future_y, future_yaw], dim=-1)
        
        return poses
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # image, bev, agent, traj, score
        camera_feature: torch.Tensor = features["image"]
        status_feature: torch.Tensor = features["ego_status"]
        batch_size = status_feature.shape[0]
        constant_velocity = self.compute_constant_velocity(status_feature)

        scene_token = self.scene_embeds.repeat(batch_size, 1, 1, 1)

        img_token = self.image_backbone(camera_feature, scene_token)
        # img_token = torch.randn(batch_size, 
        #                         self._config.num_cams * self._config.num_scene_tokens, 
        #                         self._config.tf_d_model)
        
        ego_token = self.hist_encoding(status_feature)
        ego_token = ego_token + self.hist_pos_embed

        trajectory = self._trajectory_head(constant_velocity, img_token, ego_token)

        output = self._score_head(trajectory, img_token, ego_token)

        output['img_token'] = img_token
        output['ego_token'] = ego_token
        output['constant_velocity'] = constant_velocity

        return output

