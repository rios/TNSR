#!/bin/bash

ns-train tnsr-refine \
    --project-name tnsr-ball \
    --experiment-name tnsr-ball-refine \
    --load-checkpoint ./outputs/tnsr-ball-initial/tnsr-initial/2024-06-13_181248/nerfstudio_models/step-000040000.ckpt \
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
    --pipeline.model.near-plane 0.05 \
    --pipeline.model.far-plane 20 \
    --pipeline.model.overwrite_near_far_plane True \
    --pipeline.model.refinement True \
    --pipeline.model.cuboid_coordinates -0.14 -0.11 -0.06 0.10 0.13 0.15 \
    --pipeline.model.eta 1.48 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.model.background-model mlp \
    --vis wandb \
    --data /home/weijian/Workspace/Nerfdatasets/Ball/  nerfstudio-data --center_method focus
