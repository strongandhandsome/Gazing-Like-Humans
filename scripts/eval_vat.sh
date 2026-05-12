#!/bin/bash
# Evaluate a per-frame (no temporal attention) model on VideoAttentionTarget.
# This script is kept for users who have their own static-mode VAT weights;
# the released VAT checkpoint corresponds to the temporal model — use
# eval_vattemp.sh for it.

DATA_PATH=${DATA_PATH:-./data/videoattentiontarget}
CKPT=${CKPT:-./saved_weights/vitl/vat/weight.pt}

python scripts/eval_vat.py \
    --data_path "$DATA_PATH" \
    --model gazelle_ms_dinov2_vitl14_inout \
    --ckpt "$CKPT" \
    --batch_size 64
