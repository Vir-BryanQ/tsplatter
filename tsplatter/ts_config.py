from __future__ import annotations

from tsplatter.data.thermalmap_dataparser import ThermalMapDataParserConfig
from tsplatter.ts_datamanager import TSplatterManagerConfig
from tsplatter.ts_model import TSplatterModelConfig
from tsplatter.ts_pipeline import TSplatterPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

tsplatter = MethodSpecification(
    config=TrainerConfig(
        method_name="tsplatter",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},

        pipeline=TSplatterPipelineConfig(
            datamanager=TSplatterManagerConfig(
                dataparser=ThermalMapDataParserConfig()
            ),
            model=TSplatterModelConfig(),
        ),
        
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6, max_steps=30000
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            # "means": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-20, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(
            #         lr_final=1e-20, max_steps=30000
            #     ),
            # },
            # "opacities": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-20, eps=1e-15),
            #     "scheduler": None,
            # },
            # "scales": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-20, eps=1e-15),
            #     "scheduler": None,
            # },
            # "quats": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-20, eps=1e-15),
            #     "scheduler": None,
            # },

            "thermal_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "thermal_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="TSplatter: quick thermal splatter",
)