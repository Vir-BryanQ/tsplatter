import json
import os
import random
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
import trimesh
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    DDP,
    Model,
    VanillaPipeline,
    VanillaPipelineConfig,
    dist,
)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE
from ts_model import TSplatterModelConfig
from ts_datamanager import TSplatterManagerConfig


@dataclass
class TSplatterPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: TSplatterPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=TSplatterManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=TSplatterModelConfig)
    """specifies the model config"""

class TSplatterPipeline(VanillaPipeline):
    def __init__(
        self,
        config: DNSplatterPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            train_dataset=self.datamanager.train_dataset
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                Model,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])