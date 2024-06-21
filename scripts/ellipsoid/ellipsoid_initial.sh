#!/bin/bash

ns-train tnsr-initial \
     --project-name nsr-ellipsoid \
     --experiment-name nsr-ellipsoid-initial \
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
     --pipeline.model.overwrite_near_far-plane True \
     --pipeline.model.smooth_loss_multi 0.05 \
     --pipeline.model.background-model none \
     --pipeline.datamanager.train-num-rays-per-batch 2048 \
     --vis wandb \
     --data /YOUR_PATH/ellipsoid blender-data \
     --scale_factor 0.1
