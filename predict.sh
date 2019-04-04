#!/bin/bash
SIDEWALK_DETECTOR_WEIGHT_PATH=./resources/segmentation/weight_binary.pth
SURFACE_SEGMENTATOR_WEIGHT_PATH=./resources/segmentation/weight_multi.pth
python3 eval_two_stage_seg.py\
    --sidewalk-detector-weight-path $SIDEWALK_DETECTOR_WEIGHT_PATH\
    --surface-segmentator-weight-path $SURFACE_SEGMENTATOR_WEIGHT_PATH\
    --image-paths $1\
    --save-path ./out\
    --seed 1\
    --input-size 640 640\
