#!/bin/bash

ns-train tnsr-initial \
    --project-name tnsr-glass \
    --experiment-name tnsr-glass-initial \
    --pipeline.model.sdf-field.use-grid-feature True \
    --pipeline.model.sdf-field.hidden-dim 256 \
    --pipeline.model.sdf-field.num-layers 2 \
    --pipeline.model.sdf-field.num-layers-color 2 \
    --pipeline.model.sdf-field.use-appearance-embedding False \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.inside-outside False \
    --pipeline.model.sdf-field.bias 0.5 \
    --pipeline.model.sdf-field.beta-init 0.3 \
    --pipeline.model.smooth_loss_multi 0.05 \
    --pipeline.model.near-plane 0.01 \
    --pipeline.model.far-plane 20 \
    --pipeline.model.overwrite_near_far_plane True \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.model.background-model mlp \
    --vis wandb \
    --data /home/weijian/Workspace/Nerfdatasets/glass/  nerfstudio-data --center_method focus
