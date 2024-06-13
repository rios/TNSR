#!/bin/bash

ns-train tnsr-refine \
     --load-checkpoint ./outputs/nsr-waterball-initial/tnsr-initial/2024-05-29_173600/nerfstudio_models/step-000040000.ckpt \
     --project-name nsr-optical \
     --experiment-name nsr-optical-refine \
     --pipeline.model.sdf-field.use-grid-feature True \
     --pipeline.model.sdf-field.hidden-dim 256 \
     --pipeline.model.sdf-field.num-layers 2 \
     --pipeline.model.sdf-field.num-layers-color 2 \
     --pipeline.model.sdf-field.use-appearance-embedding False \
     --pipeline.model.sdf-field.geometric-init True \
     --pipeline.model.sdf-field.inside-outside False \
     --pipeline.model.sdf-field.bias 0.5 \
     --pipeline.model.sdf-field.beta-init 0.3 \
     --pipeline.model.near-plane 0.2 \
     --pipeline.model.far-plane 10 \
     --pipeline.model.overwrite_near_far_plane True \
     --pipeline.model.smooth_loss_multi 0.02 \
     --pipeline.model.background-model none \
     --pipeline.model.refinement True \
     --pipeline.model.cuboid_coordinates -0.15 -0.15 -0.15 0.15 0.15 0.15 \
     --pipeline.model.eta 1.45 \
     --pipeline.datamanager.train-num-rays-per-batch 2048 \
     --vis wandb \
     --data /home/weijian/Workspace/Nerfdatasets/optical blender-data \
     --scale_factor 0.1
