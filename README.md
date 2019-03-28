# Sidewalk Semantic Segmentation

![logo](./road_roughness_prediction/app/static/logo.png)

[![Build Status](https://travis-ci.com/mknz/dsr-road-roughness-prediction.svg?branch=master)](https://travis-ci.com/mknz/dsr-road-roughness-prediction)

## Live demo

[here](http://sidewalk.online/)

## Run all tests

```
python -m pytest tests
```

## Run interactive tests (This shows some images)

```
python -m pytest tests -m interactive --interactive
```

## See training logs using tensorboard

```
tensorboard --logdir $LOGDIR
```
