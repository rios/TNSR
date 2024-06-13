#!/bin/bash

ns-train tnsr-refine \
    --project-name tnsr-glass \
    --experiment-name tnsr-glass-refine \
    --load-checkpoint ./outputs/tnsr-glass-initial/tnsr-initial/2024-05-31_003105/nerfstudio_models/step-000040000.ckpt \
    --pipeline.model.sdf-field.use-grid-feature True \
    --pipeline.model.sdf-field.hidden-dim 256 \
    --pipeline.model.sdf-field.num-layers 2 \
    --pipeline.model.sdf-field.num-layers-color 2 \
    --pipeline.model.sdf-field.use-appearance-embedding False \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.inside-outside False \
    --pipeline.model.sdf-field.bias 0.5 \
    --pipeline.model.sdf-field.beta-init 0.3 \
    --pipeline.model.smooth_loss_multi 0.02 \
    --pipeline.model.near-plane 0.01 \
    --pipeline.model.far-plane 20 \
    --pipeline.model.overwrite_near_far_plane True \
    --pipeline.model.refinement True \
    --pipeline.model.cuboid_coordinates -0.15 -0.15 -0.15 0.20 0.16 0.40 \
    --pipeline.model.eta 1.32 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.model.background-model mlp \
    --pipeline.model.refinement True \
    --vis wandb \
    --data /home/weijian/Workspace/Nerfdatasets/glass/  nerfstudio-data --center_method focus
