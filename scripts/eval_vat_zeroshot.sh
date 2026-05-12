#!/bin/bash
# Zero-shot evaluation on VideoAttentionTarget using GazeFollow-trained weights.

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/weight.pt}

python scripts/eval_vat_zeroshot.py \
    --data_path "$DATA_PATH" \
    --model GazeFollow_glh_vitl14 \
    --ckpt "$CKPT" \
    --batch_size 64 \
    --split test
