import os
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.flow.flow_config import FlowConfig
from navsim.agents.flow.flow_model import FlowModel, FlowHead
from navsim.agents.flow.flow_features import FlowFeatureBuilder, FlowTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

import uuid
import logging
import traceback
import pandas as pd
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataloader import MetricCacheLoader
from pathlib import Path
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMAgents
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from navsim.common.dataclasses import PDMResults
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from functools import partial
from navsim.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)

def run_pdm_score(data_batch: List[Dict], metric_cache_loader, simulator, scorer, reactive_policy):
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    pdm_results: List[pd.DataFrame] = []
    tokens = [a['token'] for a in data_batch]
    pred_trajectory = [a['pred_trajectory'] for a in data_batch]

    for idx, (token) in enumerate(tokens):
        logger.info(
            f"Processing stage one reactive scenario {idx + 1} / {len(tokens)} in thread_id={thread_id}, node_id={node_id}"
        )
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = pred_trajectory[idx]

            score_row_stage_one, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=reactive_policy,
            )
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row_stage_one = pd.DataFrame([PDMResults.get_empty_results()])
            score_row_stage_one["valid"] = False
        score_row_stage_one["token"] = token

        pdm_results.append(score_row_stage_one)
    
    return pdm_results

class FlowAgent(AbstractAgent):
    """Agent interface for Flow baseline."""

    def __init__(
            self,
            config: FlowConfig,
            lr: float,
            checkpoint_path: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self.model = FlowModel(config)

        self.worker = kwargs.get('worker')
        self.metric_cache_loader = MetricCacheLoader(Path(kwargs.get('metric_cache_path')))
        self.simulator: PDMSimulator = kwargs.get('simulator')
        self.scorer: PDMScorer = kwargs.get('scorer')
        assert (
        self.simulator.proposal_sampling == self.scorer.proposal_sampling
        ), "Simulator and scorer proposal sampling has to be identical"

        self.reactive_policy: NavsimIDMAgents = kwargs.get('reactive_policy')

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: Transfuser only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=history_steps,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [FlowTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [FlowFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)
    
    def get_pred_traj_pdm_score(self, pred_trajectory, tokens):
        data_points = [
            {'token': token, 'pred_trajectory': Trajectory(pred, self._config.trajectory_sampling)} 
            for token, pred in zip(tokens, pred_trajectory)
        ]

        worker_fn = partial(
            run_pdm_score, 
            metric_cache_loader=self.metric_cache_loader, 
            simulator=self.simulator, 
            scorer=self.scorer, 
            reactive_policy=self.reactive_policy
        )

        score_rows: List[pd.DataFrame] = worker_map(self.worker, worker_fn, data_points)
        
        pdm_score_df = pd.concat(score_rows)

        pdm_score_df.set_index('token', inplace=True)

        score_cols = [
        'no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','traffic_light_compliance',
        'time_to_collision_within_bound','lane_keeping','history_comfort','ego_progress']

        sorted_df = pdm_score_df.loc[tokens, score_cols]
        tensor_df = torch.from_numpy(sorted_df.values).float()  # (B, 8)

        # 2. 计算样本权重 (Sample Weights)
        # 检查前 4 项 (乘性因子)，只要有一个为 0，就是坏样本
        multiplier_metrics = tensor_df[:, :4] 
        # ~torch.any(... > 0) 等同于 "只要有一个是 0"
        is_bad_sample = (~torch.any(multiplier_metrics > 0, dim=1)).float()
        
        # 动态调整惩罚力度
        bad_ratio = is_bad_sample.mean().item()
        if bad_ratio > 0.3:
            penalty_factor = 3.0   # 坏样本多，温和惩罚
        elif bad_ratio > 0.1:
            penalty_factor = 8.0   # 正常情况
        else:
            penalty_factor = 20.0  # 坏样本少，严厉挖掘
            
        sample_weights = 1.0 + penalty_factor * is_bad_sample

        return tensor_df, sample_weights
    
    def get_score_loss(self, mode, pred_logits, tensor_df, sample_weights=None):
        # --- 计算 Loss ---
        loss_dict = {}
        total_loss = 0.0
        device = pred_logits.device # 获取预测值所在的设备 (GPU)
        tensor_df = tensor_df.to(device) # 确保 tensor_df 在同一设备上
        tensor_df[:, 0][tensor_df[:, 0] == 0.5] = 0.0
        tensor_df[:, 2][tensor_df[:, 2] == 0.5] = 0.0
        
        if sample_weights is None:
            sample_weights = torch.ones(tensor_df.shape[0], device=device)
        else:
            sample_weights = sample_weights.to(device)
        
        metric_configs = [
            ('nc', 0, 'bce', 20.0), ('dac', 1, 'bce', 20.0),
            ('ddc', 2, 'bce', 20.0), ('tlc', 3, 'bce', 20.0),
            ('ttc', 4, 'bce', 1.0), ('lk', 5, 'bce', 1.0),
            ('hc', 6, 'bce', 1.0), ('ep', -1, 'mse', 1.0)
        ]

        # GT 模式下的特殊处理：EC 指标需要掩码过滤 NaN
        if mode == 'gt':
            metric_configs.append(('ec', -2, 'bce_masked', 1.0))

        # ==========================================
        # 统一循环计算 Loss
        # ==========================================
        for name, idx, loss_type, weight in metric_configs:
            pred = pred_logits[:, idx]
            target = tensor_df[:, idx]

            # 根据类型选择损失函数
            if loss_type == 'mse':
                raw_loss = F.mse_loss(pred, target, reduction='none')
                current_weights = sample_weights
            elif loss_type == 'bce_masked':
                # 处理 GT 模式下 EC 的 NaN 值
                mask = ~torch.isnan(target)
                if mask.sum() == 0: continue # 如果没有有效值则跳过
                raw_loss = F.binary_cross_entropy_with_logits(pred[mask], target[mask], reduction='none')
                current_weights = sample_weights[mask]
            else: # 默认 bce
                raw_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
                current_weights = sample_weights

            # 统一应用样本权重并归一化
            weighted_loss = raw_loss * current_weights
            loss_val = weighted_loss.sum() / current_weights.sum()

            # 记录字典并累加总 Loss
            loss_dict[f'{mode}_{name}_loss'] = raw_loss.float().mean()
            total_loss += weight * loss_val

        return total_loss, loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens
    ):
        tensor_df, sample_weights = self.get_pred_traj_pdm_score(predictions['trajectory'].detach().cpu().numpy(), tokens)
        
        pred_score_loss, pred_score_loss_dict = self.get_score_loss(
            'pred', 
            predictions['pred_logits'], 
            tensor_df, 
            sample_weights=sample_weights
        )

        gt_predictions = self.model._score_head.forward(targets['trajectory'].float(), 
                                       predictions['img_token'], 
                                       predictions['ego_token'])
        gt_score_loss, gt_score_loss_dict = self.get_score_loss(
            'gt',gt_predictions['pred_logits'], targets['gt_score'])
        
        flow_loss = self.model._flow_head.get_flow_loss(targets, predictions)

        gt_expanded = targets['trajectory'].unsqueeze(1)
        dists = torch.norm(predictions['flow_proposal'] - gt_expanded, dim=-1, p=1).mean(dim=-1)
        mindist_loss = torch.min(dists, dim=1)[0].mean()
        
        # ec_pred_logit = predictions['pred_logits'][:, -2]

        # bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        
        # dp_loss = dp_loss * config.dp_loss_weight
        # bev_semantic_loss = bev_semantic_loss * config.bev_loss_weight
        total_loss = (
                flow_loss +
                mindist_loss +
                gt_score_loss +
                pred_score_loss
                # bev_semantic_loss
        )

        total_loss = total_loss.float()

        # 构建返回字典
        return total_loss, {
            'total_loss': total_loss,
            'max_scores': torch.mean(predictions['max_scores']).float(),
            'flow_loss': flow_loss.float(),
            'mindist_loss': mindist_loss.float(),
            'gt_score_loss': gt_score_loss.float(),
            **gt_score_loss_dict,
            'pred_score_loss': pred_score_loss.float(),
            **pred_score_loss_dict,
        }

    def get_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._lr)

        # T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)
        T_max = 10000

        # classic cosine
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=T_max, 
        #     eta_min=0.0, last_epoch=-1
        # )

        # Ramp + cosine
        T_max_ramp = int(T_max * 0.1)
        scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
        T_max_cosine = T_max - T_max_ramp
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max_cosine, 
            eta_min=0.0, last_epoch=-1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_ramp, scheduler_cosine],
            milestones=[T_max_ramp],
        )           

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
    def get_training_callbacks(self) -> List[pl.Callback]:
        ckpt_callback_best = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="val/max_scores",
            mode="max",
            dirpath=f"/root/ckpt/",
            filename="best-{epoch:02d}-{step:04d}"
        )

        return [
            ckpt_callback_best
        ]
