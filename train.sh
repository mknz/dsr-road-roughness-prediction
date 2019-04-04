#!/bin/bash
DATASET_NAME=

TRAIN_IMG_PATH=
TRAIN_MASK_PATH=
VALID_IMG_PATH=
VALID_MASK_PATH=

JACCARD_WEIGHT=0.2
INPUT_SIZE=640
EPOCHS=100
BATCH_SIZE=8

python3 train_seg.py\
    --train-image-dirs\
        $TRAIN_IMG_PATH\
    --train-mask-dirs\
        $TRAIN_MASK_PATH\
    --train-dataset-types\
        base\
    --validation-image-dirs\
        $VALID_IMG_PATH\
    --validation-mask-dirs\
        $VALID_MASK_PATH\
    --validation-dataset-types\
        base\
    --category-type simple\
    --batch-size $BATCH_SIZE\
    --epochs $EPOCHS\
    --model-name unet11\
    --input-size $INPUT_SIZE $INPUT_SIZE\
    --jaccard-weight $JACCARD_WEIGHT\
    --run-name ${INPUT_SIZE}x${INPUT_SIZE}_jaccard_${JACCARD_WEIGHT}_batch_$BATCH_SIZE\
    --save-dir ./runs/simple/$DATASET_NAME\
