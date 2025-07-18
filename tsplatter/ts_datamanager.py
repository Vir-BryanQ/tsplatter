import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import torchvision.transforms.functional as TF

from tsplatter.data.ts_dataset import TSDataset
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset

@dataclass
class TSplatterManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: TSplatterDataManager)

class TSplatterDataManager(FullImageDatamanager):
    config: TSplatterManagerConfig
    train_dataset: TSDataset
    eval_dataset: TSDataset

    def __init__(
        self,
        config: TSplatterManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return TSDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return TSDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(
                split=self.test_split
            ),
            scale_factor=self.config.camera_res_scale_factor,
        )