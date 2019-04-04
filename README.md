# Sidewalk Semantic Segmentation

[![Build Status](https://travis-ci.com/mknz/dsr-road-roughness-prediction.svg?branch=master)](https://travis-ci.com/mknz/dsr-road-roughness-prediction)

![logo](./road_roughness_prediction/app/static/logo.png)

## Live demo

[here](http://sidewalk.online/)

## About

This is the repository of our portfolio project in [DSR](https://www.datascienceretreat.com/). In this project, we built a classifier that can predict the surface category of sidewalks in Berlin from street view images. We developed this as a basis for the future routing application that can tell users road conditions.

### Contributors

- [Masanori Kanazu](https://github.com/mknz)
- [Dmitry Yefimenko](https://github.com/Dyefimenko)

## Models

The basis model is [U-Net with VGG11 encoder pretrained with ImageNet](https://github.com/ternaus/TernausNet).

We trained two models, one is a sidewalk detector, and the other is a surface category classifier. The two outputs were combined to make a final prediction.

## Data

We collected 798 images from Google Street View and manually annotated them. We supplemented the dataset with 77 photos that we took ourselves. For the sidewalk detector, we further supplemented the dataset with 1424 images from [Berkeley DeepDrive](https://bdd-data.berkeley.edu/) dataset.

We used 30 GSV images for the validation, and other 30 GSV images for the test.

## Results

### Binary classification test IoU

| Category           | IoU    |
| ------------------ |:-------|
| Sidewalk           |  0.838 |

### Surface category classification test IoU

| Category           | IoU    |
| ------------------ |:-------|
| Flat Stones        |  0.547 |
| Pavement Stone     |  0.462 |
| Sett               |  0.602 |
| Bicycle Tiles      |  0.419 |

### Segmentation examples

![example1](./road_roughness_prediction/app/static/segmentation/00012.jpg)
![example2](./road_roughness_prediction/app/static/segmentation/00023.jpg)

## Development

### Install dependencies

```
pip3 install -r requirements.txt
```

### Predict an image

#### Download trained weights

```
./donwload_weights.sh
```

#### Run a prediction

```
./predict.sh ./tests/resources/segmentation/labelme/JPEGImages/zyZ1BD8DoUJ2.jpg
```

This outputs segmented images to `./out`

### Train model

```
./train.sh
```

### Run all tests

```
python -m pytest tests
```

### Run interactive tests (This shows some images)

```
python -m pytest tests -m interactive --interactive
```

### See training logs using tensorboard

```
tensorboard --logdir $LOGDIR
```

## License

[MIT](https://raw.githubusercontent.com/mknz/dsr-road-roughness-prediction/master/LICENSE)
