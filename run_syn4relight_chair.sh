#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=1 python train_refgaussian.py \
    -s ~/data/Synthetic4Relight/chair \
    -m outputs/Synthetic4Relight/chair/refgs \
    --eval -w --lambda_mask_entropy 0.05

CUDA_VISIBLE_DEVICES=1 python train.py \
    -s ~/data/Synthetic4Relight/chair \
    --eval --iterations 20000 \
    --start_checkpoint_refgs outputs/Synthetic4Relight/chair/refgs/chkpnt50000.pth \
    --envmap_resolution 128 \
    --lambda_base_color_smooth 2 \
    --lambda_roughness_smooth 2 \
    --diffuse_sample_num 256 \
    --envmap_cubemap_lr 0.01 \
    --lambda_light_smooth 0.0005 \
    --init_roughness_value 0.6 \
    --lambda_light 0.01 \
    -m outputs/Synthetic4Relight/chair/irgs --train_ray

CUDA_VISIBLE_DEVICES=1 python render.py \
    -m outputs/Synthetic4Relight/chair/irgs \
    --eval --diffuse_sample_num 512 \
    --no_save --no_lpips
CUDA_VISIBLE_DEVICES=1 python compute_albedo_scale_syn4.py \
    -m outputs/Synthetic4Relight/chair/irgs
CUDA_VISIBLE_DEVICES=1 python eval_material_syn4.py \
    -m outputs/Synthetic4Relight/chair/irgs \
    --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=1 python eval_relighting_syn4.py \
    -m outputs/Synthetic4Relight/chair/irgs \
    --diffuse_sample_num 512 \
    --light_sample_num 256 \
    --albedo_rescale 2 \
    --no_save --no_lpips -e light
