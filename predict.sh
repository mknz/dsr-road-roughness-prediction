#!/bin/bash
BINARY_WEIGHT_PATH=./resources/segmentation/weight_binary.pth
MULTI_WEIGHT_PATH=./resources/segmentation/weight_multi.pth
python3 eval_two_stage_seg.py\
    --sidewalk-detector-weight-path $BINARY_WEIGHT_PATH\
    --surface-segmentator-weight-path $MULTI_WEIGHT_PATH\
    --image-paths $1\
    --save-path ./out\
    --seed 1\
    --input-size 640 640\
