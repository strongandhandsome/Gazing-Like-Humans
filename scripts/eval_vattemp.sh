#!/bin/bash
# Evaluate the multi-scale + temporal ViT-L model on the VideoAttentionTarget
# test split. Predictions are taken at the middle frame of each clip.

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
CKPT=${CKPT:-./saved_weights/vitl/vat/weight.pt}

python scripts/eval_vattemp.py \
    --data_path "$DATA_PATH" \
    --model VAT_glh_vitl14 \
    --ckpt "$CKPT" \
    --batch_size 32 \
    --clip_length 7 \
    --sample_rate 5
