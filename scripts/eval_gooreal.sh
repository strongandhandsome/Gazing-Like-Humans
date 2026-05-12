#!/bin/bash
# Cross-domain evaluation on GOO-Real using GazeFollow-trained weights.

DATA_PATH=${DATA_PATH:-./data/goo-real}
CKPT=${CKPT:-./saved_weights/vitl/gazefollow/weight.pt}

python scripts/eval_gooreal.py \
    --data_path "$DATA_PATH" \
    --model GazeFollow_glh_vitl14 \
    --ckpt "$CKPT" \
    --batch_size 64
