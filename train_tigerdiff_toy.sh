#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

python3 main.py \
    --model=TigerDiff \
    --dataset=AmazonReviews2014 \
    --category="${CATEGORY:-Toys_and_Games}" \
    --run_id=tigerdiff_toy \
    --lr=0.0005 \
    --weight_decay=0.01 \
    --train_batch_size=256 \
    --eval_batch_size=64 \
    --sent_emb_batch_size=256 \
    --max_item_seq_len=50 \
    --n_embd=448 \
    --n_layer=4 \
    --n_head=4 \
    --n_inner=1024 \
    --embd_pdrop=0.1 \
    --attn_pdrop=0.1 \
    --diffloss_w=512 \
    --diffloss_d=3 \
    --num_sampling_steps=20 \
    --diffusion_batch_mul=1 \
    --diff_temperature=1.0 \
    --tiger_uncond_prob=0.1 \
    --tiger_guidance_weight=2.0 \
    --tiger_knn_metric=cosine \
    --use_rectified_flow=True \
    --rectified_flow_steps=1000 \
    --ode_solver=euler \
    --temperature=0.07 \
    "$@"
