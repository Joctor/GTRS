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
            score_row_stage_one["valid"] = True
            score_row_stage_one["log_name"] = metric_cache.log_name
            score_row_stage_one["frame_type"] = metric_cache.scene_type
            score_row_stage_one["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row_stage_one["endpoint_x"] = absolute_endpoint.x
            score_row_stage_one["endpoint_y"] = absolute_endpoint.y
            score_row_stage_one["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row_stage_one["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row_stage_one["ego_simulated_states"] = [ego_simulated_states]  # used for two-frames extended comfort

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

        return pdm_score_df
    
    def get_score_loss(self, mode, pred_logits, tensor_df):
        # --- 计算 Loss ---
        loss_dict = {}
        total_loss = 0.0
        device = pred_logits.device # 获取预测值所在的设备 (GPU)
        tensor_df = tensor_df.to(device) # 确保 tensor_df 在同一设备上
        
        bce_metric = ['nc', 'dac', 'ddc', 'tlc','ttc', 'lk', 'hc']
        for i, name in enumerate(bce_metric):
            loss = F.binary_cross_entropy_with_logits(pred_logits[:, i], tensor_df[:, i])
            loss_dict[f'{mode}_{name}_loss'] = loss.item()
            total_loss += loss
        
        # ec
        if mode == 'gt':
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
        
        pdm_score_df = self.get_pred_traj_pdm_score(predictions['trajectory'].cpu().numpy(), tokens)

        gt_score_loss, gt_score_loss_dict = self.get_score_loss(
            'gt',gt_predictions['pred_logits'], targets['gt_score'])
        
        # pred ec 全局计算
        score_cols = [
        'no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','traffic_light_compliance',
        'time_to_collision_within_bound','lane_keeping','history_comfort','ego_progress']

        sorted_df = pdm_score_df.loc[tokens, score_cols]
        tensor_df = torch.from_numpy(sorted_df.values).float()  # (B, 8)
        
        pred_score_loss, pred_score_loss_dict = self.get_score_loss(
            'pred',predictions['pred_logits'], tensor_df)
        
        ec_pred_logit = predictions['pred_logits'][:, -2]
        
        # bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
        
        # dp_loss = dp_loss * config.dp_loss_weight
        # bev_semantic_loss = bev_semantic_loss * config.bev_loss_weight
        total_loss = (
                flow_loss +
                gt_score_loss +
                pred_score_loss
                # bev_semantic_loss
        )
        return total_loss, {
            'total_loss': total_loss,
            'log_epdms_score':torch.mean(predictions['log_epdms_score']).item(),
            'flow_loss': flow_loss,
            'gt_score_loss': gt_score_loss.item(),
            **gt_score_loss_dict,
            'pred_score_loss': pred_score_loss.item(),
            **pred_score_loss_dict,
            # 'bev_semantic_loss': bev_semantic_loss
        }, pdm_score_df, ec_pred_logit

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
