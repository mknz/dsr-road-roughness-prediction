#!/bin/bash

SAVE_DIR=/tmp/road_oughness_prediction/tests/train
if [ -d $SAVE_DIR ]; then
    echo "Clean up $SAVE_DIR"
    rm -rf $SAVE_DIR
fi

for MODEL in resnet18 tiny_cnn; do
    echo $MODEL
    python3 scripts/train.py\
        --data-dir ./tests/resources/surfaces\
        --categories asphalt grass\
        --target-dir-name ready\
        --batch-size 128\
        --epochs 1\
        --model-name $MODEL\
        --class-balanced\
        --save-dir $SAVE_DIR
    if [ $? -ne 0 ]; then
        echo "$MODEL training failed"
        exit 1
    fi
done
