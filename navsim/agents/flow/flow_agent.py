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

def get_score_loss(preds, targets):
    probs_mult = preds['pred_mult']       # (B, K, 4)
    probs_weighted = preds['pred_weighted']     # (B, K, 5)

    gt_mult = targets['gt_mult']             # (B, K, 4) 或 (B, 4)
    gt_weighted = targets['gt_weighted']     # (B, K, 5) 或 (B, 5)

    gt_mult = gt_mult.expand(-1, probs_mult.shape[1], -1)
    gt_weighted = gt_weighted.expand(-1, probs_weighted.shape[1], -1)

    # 确保设备一致
    device = probs_mult.device
    gt_mult = gt_mult.to(device)
    gt_weighted = gt_weighted.to(device)
    
    # --- 计算 Loss ---
    loss_dict = {}
    total_loss = 0.0
    mult_names = ['nc', 'dac', 'ddc', 'tlc']
    for i, name in enumerate(mult_names):
        # 单独计算第 i 列的 Loss
        l = F.binary_cross_entropy_with_logits(probs_mult[:, :, i], gt_mult[:, :, i])
        loss_dict[f'{name}_loss'] = l.item() # 记录数值用于日志
        total_loss += l * 2.0  # 累加到总 Loss (乘以权重)
        
    # --- 2. 单独计算 Weighted 的每一项 ---
    weighted_names = ['ttc', 'ep', 'lk', 'hc', 'ec']
    pred_scores = torch.sigmoid(probs_weighted)
    for i, name in enumerate(weighted_names):
        l = F.mse_loss(pred_scores[:, :, i], gt_weighted[:, :, i])
        loss_dict[f'{name}_loss'] = l.item() # 记录数值
        total_loss += l * 1.0  # 累加到总 Loss
    
    return total_loss, loss_dict

def flow_loss_bev(
        features,
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor],
        tokens: List[str], config: FlowConfig, traj_head: FlowHead
):
    # B, 8 (4 secs, 0.5Hz), 3
    target_traj = targets["trajectory"]

    flow_loss = traj_head.get_flow_loss(predictions['env_kv'], target_traj.float())
    score_loss, score_loss_dict = get_score_loss(predictions, targets)
    # bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    
    # dp_loss = dp_loss * config.dp_loss_weight
    # bev_semantic_loss = bev_semantic_loss * config.bev_loss_weight
    total_loss = (
            flow_loss +
            score_loss
            # bev_semantic_loss
    )
    return total_loss, {
        'total_loss': total_loss,
        'flow_loss': flow_loss,
        'score_loss': score_loss.item(),
        **score_loss_dict,
        # 'bev_semantic_loss': bev_semantic_loss
    }

class FlowAgent(AbstractAgent):
    """Agent interface for Flow baseline."""

    def __init__(
            self,
            config: FlowConfig,
            lr: float,
            checkpoint_path: Optional[str] = None,
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

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens=None
    ):

        return flow_loss_bev(features, targets, predictions, tokens, self._config, self.model._trajectory_head)

    
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
            monitor="train/total_loss_epoch",
            mode="min",
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
            filename="{epoch:02d}-{step:04d}"
        )

        return [
            ckpt_callback_best
        ]
