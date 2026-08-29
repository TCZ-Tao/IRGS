#!/bin/bash
# Single TensoIR scene. Override via env, e.g.:
#   SCENE=lego GPU=7 bash run_tensoir.sh
set -euo pipefail
cd "$(dirname "$0")"

SCENE=${SCENE:-armadillo}
GPU=${GPU:-4}
DATA_ROOT=${DATA_ROOT:-$HOME/data/TensoIR_Synthetic}
DATA=${DATA:-${DATA_ROOT}/${SCENE}}
OUT=${OUT:-outputs/TensoIR_Synthetic/${SCENE}}

LAMBDA_ROUGHNESS_SMOOTH=2
LAMBDA_LIGHT_SMOOTH=0.0005
INIT_ROUGHNESS_VALUE=0.6
LAMBDA_LIGHT=0.1
EXTRA_TRAIN_ARGS=()

case "${SCENE}" in
    hotdog)
        EXTRA_TRAIN_ARGS=(--light_t_min 0.05)
        ;;
    lego)
        LAMBDA_ROUGHNESS_SMOOTH=0.1
        LAMBDA_LIGHT_SMOOTH=0.05
        INIT_ROUGHNESS_VALUE=0.8
        LAMBDA_LIGHT=0.5
        ;;
esac

# ${SCENE} on GPU ${GPU}
CUDA_VISIBLE_DEVICES=${GPU} python train_refgaussian.py \
    -s ${DATA} \
    -m ${OUT}/refgs \
    --eval -w --lambda_mask_entropy 0.05

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    -s ${DATA} \
    --eval --iterations 20000 \
    --start_checkpoint_refgs ${OUT}/refgs/chkpnt50000.pth \
    --envmap_resolution 128 \
    --lambda_base_color_smooth 2 \
    --lambda_roughness_smooth ${LAMBDA_ROUGHNESS_SMOOTH} \
    --diffuse_sample_num 256 \
    --envmap_cubemap_lr 0.01 \
    --lambda_light_smooth ${LAMBDA_LIGHT_SMOOTH} \
    --init_roughness_value ${INIT_ROUGHNESS_VALUE} \
    --lambda_light ${LAMBDA_LIGHT} \
    ${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"} \
    -m ${OUT}/irgs --train_ray

CUDA_VISIBLE_DEVICES=${GPU} python render.py \
    -m ${OUT}/irgs \
    --eval --diffuse_sample_num 512 \
    --no_save --no_lpips
CUDA_VISIBLE_DEVICES=${GPU} python compute_albedo_scale_tensoir.py \
    -m ${OUT}/irgs
CUDA_VISIBLE_DEVICES=${GPU} python eval_material_tensoir.py \
    -m ${OUT}/irgs \
    --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=${GPU} python eval_relighting_tensoir.py \
    -m ${OUT}/irgs \
    --diffuse_sample_num 512 \
    --light_sample_num 256 \
    --albedo_rescale 2 -e light
