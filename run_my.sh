SCENE=${SCENE:-dining_chair}
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-/ssd2/tcz/data/my}
DATA=${DATA:-${DATA_ROOT}/${SCENE}}
OUT=${OUT:-outputs/${SCENE}}

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
    --lambda_roughness_smooth 2 \
    --diffuse_sample_num 256 \
    --envmap_cubemap_lr 0.01 \
    --lambda_light_smooth 0.0005 \
    --init_roughness_value 0.6 \
    --lambda_light 0.01 \
    -m ${OUT}/irgs --train_ray

CUDA_VISIBLE_DEVICES=${GPU} python render.py \
    -m ${OUT}/irgs \
    --eval --diffuse_sample_num 512 \
    --no_save --no_lpips
CUDA_VISIBLE_DEVICES=${GPU} python compute_albedo_scale_syn4.py \
    -m ${OUT}/irgs
CUDA_VISIBLE_DEVICES=${GPU} python eval_material_syn4.py \
    -m ${OUT}/irgs \
    --no_save --no_lpips --albedo_rescale 2
CUDA_VISIBLE_DEVICES=${GPU} python eval_relighting_syn4.py \
    -m ${OUT}/irgs \
    --diffuse_sample_num 512 \
    --light_sample_num 256 \
    --albedo_rescale 2 \
    --no_save --no_lpips -e light
