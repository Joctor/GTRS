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
import pandas as pd
import numpy as np

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

        # TODO
        # self.last_epoch_pdm_result = pd.read_csv(self._config.last_epoch_pdm_result_path)
        # self.last_epoch_pdm_result.set_index('token', inplace=True)
        
        # last_epoch = np.load(f"{os.environ.get('NAVSIM_DEVKIT_ROOT')}/epoch_1.npz")
        # self.last_epoch_tokens = last_epoch['tokens']
        # self.last_epoch_pred_logits = last_epoch['pred_logits']

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
    
    def get_score_loss(self, mode, pred_logits, tensor_df):
        # --- 计算 Loss ---
        loss_dict = {}
        total_loss = 0.0
        device = pred_logits.device # 获取预测值所在的设备 (GPU)
        tensor_df = tensor_df.to(device) # 确保 tensor_df 在同一设备上
        tensor_df[:, 0][tensor_df[:, 0] == 0.5] = 0.0
        tensor_df[:, 2][tensor_df[:, 2] == 0.5] = 0.0
        
        bce_metric = ['nc', 'dac', 'ddc', 'tlc','ttc', 'lk', 'hc']
        for i, name in enumerate(bce_metric):
            loss = F.binary_cross_entropy_with_logits(pred_logits[:, i], tensor_df[:, i])
            loss_dict[f'{mode}_{name}_loss'] = loss.item()
            total_loss += loss
        
        pred_col = pred_logits[:, -2]
        target_col = tensor_df[:, -2]
        # 创建掩码：非空值为 True
        mask = ~torch.isnan(target_col)
        if mask.sum() > 0:  # 确保至少有一个有效值
            loss = F.binary_cross_entropy_with_logits(pred_col[mask], target_col[mask])
            loss_dict[f'{mode}_ec_loss'] = loss.item()
            total_loss += loss

        # ep
        loss = F.mse_loss(pred_logits[:, -1], tensor_df[:, -1])
        loss_dict[f'{mode}_ep_loss'] = loss.item()
        total_loss += loss
            
        return total_loss, loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
            tokens
    ):

        flow_loss = self.model._trajectory_head.get_flow_loss(targets, predictions)

        gt_predictions = self.model._score_head.forward(targets['trajectory'].float(), 
                                       predictions['img_token'], 
                                       predictions['ego_token'])
        gt_score_loss, gt_score_loss_dict = self.get_score_loss(
            'gt',gt_predictions['pred_logits'], targets['gt_score'])

        score_cols = [
        'no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','traffic_light_compliance',
        'time_to_collision_within_bound','lane_keeping','history_comfort','two_frame_extended_comfort','ego_progress']
        
        # TODO
        # # --- 1. 准备 PDM 结果 (来自CSV) ---
        # # .loc 会强制 sorted_df 的顺序与 tokens 列表的顺序一致
        # sorted_df = self.last_epoch_pdm_result.loc[tokens, score_cols]
        # tensor_df = torch.from_numpy(sorted_df.values).float()  # (B, 9)

        # # --- 2. 准备预测 Logits (来自NPZ) ---
        # # 创建一个从 token 到其在原始数组中索引的映射
        # token_to_idx = {t: i for i, t in enumerate(self.last_epoch_tokens)}

        # # 根据当前批次 tokens 的顺序，获取它们在原始数组中的索引
        # # 这能确保 selected_logits 的顺序被强制调整为与 tokens 列表一致
        # indices = [token_to_idx[token] for token in tokens]
        # selected_logits = self.last_epoch_pred_logits[indices] # 顺序现在也与 tokens 一致
        
        # pred_score_loss, pred_score_loss_dict = self.get_score_loss(
        #     'pred',selected_logits, tensor_df)
        
        # bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        
        # dp_loss = dp_loss * config.dp_loss_weight
        # bev_semantic_loss = bev_semantic_loss * config.bev_loss_weight
        total_loss = (
                flow_loss +
                gt_score_loss
                # pred_score_loss
                # bev_semantic_loss
        )
        return total_loss, {
            'total_loss': total_loss,
            'log_epdms_score':torch.mean(predictions['log_epdms_score']).item(),
            'flow_loss': flow_loss,
            'gt_score_loss': gt_score_loss.item(),
            **gt_score_loss_dict,
            # 'pred_score_loss': pred_score_loss.item(),
            # **pred_score_loss_dict,
            # 'bev_semantic_loss': bev_semantic_loss
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
            monitor="val/log_epdms_score",
            mode="max",
            dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
            filename="best-{epoch:02d}-{step:04d}"
        )

        return [
            ckpt_callback_best
        ]
