# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import IntEnum
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import Polygon, LineString
from torchvision import transforms

#from navsim.agents.gtrs_dense.hydra_config import HydraConfig
from navsim.agents.flow.flow_config import FlowConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from PIL import Image
import pandas as pd

def state2traj(states):
    rel_poses = absolute_to_relative_poses([StateSE2(*tmp) for tmp in
                                            states[:, StateIndex.STATE_SE2]])
    final_traj = [pose.serialize() for pose in rel_poses[1:]]
    return final_traj


class FlowFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: FlowConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        data_camera = self._get_camera_feature(agent_input)
        features.update(data_camera)

        hist_es_list = []
        # include current ego status pose
        for es in agent_input.ego_statuses:
            ego_status_hist = torch.concatenate(
                [
                    torch.tensor(es.ego_pose, dtype=torch.float32),
                    torch.tensor(es.ego_velocity, dtype=torch.float32),
                    torch.tensor(es.ego_acceleration, dtype=torch.float32),
                    torch.tensor(es.driving_command, dtype=torch.float32),
                ],
            )
            hist_es_list.append(ego_status_hist)
        features["ego_status"] = torch.stack(hist_es_list)

        return features

    # from drivorR
    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # cameras = [cameras.cam_b0, cameras.cam_f0, cameras.cam_l0, cameras.cam_l1, cameras.cam_l2, cameras.cam_r0, cameras.cam_r1, cameras.cam_r2]

        # this is a change for the focus front cam
        cameras = [cameras.cam_f0, cameras.cam_b0, cameras.cam_l0, cameras.cam_l1, cameras.cam_l2, cameras.cam_r0, cameras.cam_r1, cameras.cam_r2]

        images = []
        cam_Ks = []
        lidar2cams = []
        for cam in cameras:
            if cam.image is None:
                continue

            im = Image.fromarray(cam.image)
            cam_K = np.array(cam.intrinsics)
            sensor2lidar_rotation = np.asarray(cam.sensor2lidar_rotation)
            sensor2lidar_translation = np.asarray(cam.sensor2lidar_translation)
            sensor2lidar_rt = np.eye(4)
            sensor2lidar_rt[:3, :3] = sensor2lidar_rotation
            sensor2lidar_rt[:3, 3] = sensor2lidar_translation
            lidar2cam_rt = np.linalg.inv(sensor2lidar_rt)

            # intrinsics resize
            original_size = im.size
            cam_K = cam_K.clone() if isinstance(cam_K, torch.Tensor) else cam_K.copy() # torch.Size([8, 3, 3])
            cam_K[0, 0] = cam_K[0, 0] * self._config.image_width / original_size[0]
            cam_K[1, 1] = cam_K[1, 1] * self._config.image_height / original_size[1]
            cam_K[0, 2] = cam_K[0, 2] * self._config.image_width / original_size[0]
            cam_K[1, 2] = cam_K[1, 2] * self._config.image_height / original_size[1]

            # image resize
            im = im.resize((self._config.image_width, self._config.image_height))

            # PIL to numpy and normalize
            im = np.asarray(im, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = (im - mean) / std

            # convert to torch
            im = torch.from_numpy(im).permute(2, 0, 1)
            cam_K = torch.from_numpy(cam_K)
            lidar2cam_rt = torch.from_numpy(lidar2cam_rt)

            images.append(im)
            cam_Ks.append(cam_K)
            lidar2cams.append(lidar2cam_rt)


        # Collect all camera images in a list for easier processing
        data = {
            "image": torch.stack(images),
            "cam_K": torch.stack(cam_Ks),
            "world_2_cam": torch.stack(lidar2cams)
        }
        
        return data


class FlowTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: FlowConfig):
        self._config = config
        self.pdm_df = pd.read_csv(config.pdm_result_path)
        self.pdm_df = self.pdm_df[self.pdm_df['valid'] == True]

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_traj = scene.get_future_trajectory(
            self._config.trajectory_sampling.num_poses
        )
        trajectory = torch.tensor(future_traj.poses)
        frame_idx = scene.scene_metadata.num_history_frames - 1

        gt_score = self._get_pdm_result(scene.frames[frame_idx].token)

        # annotations = scene.frames[frame_idx].annotations
        # ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        # agent_states, agent_labels = self._compute_agent_targets(annotations)
        # bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        return {
            "trajectory": trajectory,
            "gt_score": gt_score,
            # "agent_states": agent_states,
            # "agent_labels": agent_labels,
            # "bev_semantic_map": bev_semantic_map,
        }
    
    def _get_pdm_result(self, token):
        row_data = self.pdm_df[self.pdm_df['token'] == token]
        
        if row_data.empty:
            # 【修改点】不再抛出异常，而是打印警告并返回全零张量
            # 注意：在多进程环境下，print 可能不会立即显示，建议使用 logging 或写入文件
            import sys
            print(f"[WARNING] Token {token} not found in PDM results. Returning zeros.", file=sys.stderr)
            
            # 返回与正常情况形状一致的零张量
            # gt_mult shape: (1, 4), gt_weighted shape: (1, 5)
            return torch.zeros(9, dtype=torch.float32)
        
        strict_cols = ['no_at_fault_collisions', 'driving_direction_compliance']
        
        for col in strict_cols:
            if col in row_data.columns:
                row_data.loc[:, col] = row_data[col].replace(0.5, 0.0)

        mult_cols = [
        'no_at_fault_collisions',
        'drivable_area_compliance',
        'driving_direction_compliance',
        'traffic_light_compliance'
        ]

        mult_data = row_data[mult_cols].values.astype(np.float32)
        if len(mult_data) == 0:
            mult_data = np.zeros((1, 4), dtype=np.float32)
        gt_mult = torch.from_numpy(mult_data) # Shape: (N, 4)

        weighted_cols = [
        'time_to_collision_within_bound',
        'lane_keeping',
        'history_comfort',
        'two_frame_extended_comfort',
        'ego_progress',
        ]

        weighted_data = row_data[weighted_cols].values.astype(np.float32)
        if len(weighted_data) == 0:
            weighted_data = np.zeros((1, 5), dtype=np.float32)
        gt_weighted = torch.from_numpy(weighted_data) # Shape: (N, 5)

        gt_score = torch.cat([gt_mult, gt_weighted], dim=-1) # Shape: (N, 9)

        return gt_score.squeeze(0) # Shape: (9,)


    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: FlowConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                    config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
            self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _geometry_local_coords(self, geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)

    def _compute_map_polygon_mask(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
            self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw)
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
            self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects


class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
               and not attribute.startswith("__")
               and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
