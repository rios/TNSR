"""
Transparent Neural Surface Refinement implementation
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)

from nerfstudio.plugins.types import MethodSpecification

from tnsr.tnsr_field import SDFNSRFieldConfig
from tnsr.tnsr_model import NeuSNSRFactoModelConfig
from tnsr.tnsr_pipeline import TNSRPipelineConfig
from tnsr.tnsr_trainer import TNSRTrainerConfig

tnsr_initial = MethodSpecification(
    config=TNSRTrainerConfig(
        method_name="tnsr-initial",
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_all_images=10000,  # set to a very large model so we don't eval with all images
        max_num_iterations=40001,
        mixed_precision=False,
        pipeline=TNSRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
            ),
            model=NeuSNSRFactoModelConfig(
                # proposal network allows for significantly smaller sdf/color network
                sdf_field=SDFNSRFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.5,
                    beta_init=0.8,
                    use_appearance_embedding=False,
                ),
                background_model="none",
                eval_num_rays_per_chunk=2048,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(max_steps=30001, milestones=(10000, 1500, 18000)),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=40001),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=40001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="TNSR-Step1-Initial-Surface"
)

tnsr_refine = MethodSpecification(
    config=TNSRTrainerConfig(
        method_name="tnsr-refine",
        steps_per_eval_image=500,
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_all_images=5000,  # set to a very large model so we don't eval with all images
        max_num_iterations=40001,
        mixed_precision=False,
        pipeline=TNSRPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=2048,
            ),
            model=NeuSNSRFactoModelConfig(
                # proposal network allows for significantly smaller sdf/color network
                sdf_field=SDFNSRFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.5,
                    beta_init=0.8,
                    use_appearance_embedding=False,
                ),
                background_model="none",
                eval_num_rays_per_chunk=2048,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(max_steps=30001, milestones=(10000, 1500, 28000)),
            },
            "fields_color": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=00, learning_rate_alpha=0.05, max_steps=40001),
            },
            "fields_sdf": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=00, learning_rate_alpha=0.05, max_steps=40001),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=00, learning_rate_alpha=0.05, max_steps=40001),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Differentiable Neural Surface Refinement for Transparent Objects"
)
