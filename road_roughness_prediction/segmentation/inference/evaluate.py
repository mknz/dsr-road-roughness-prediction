'''Evaluation module'''
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from albumentations.augmentations.functional import center_crop

from road_roughness_prediction.segmentation import models
import road_roughness_prediction.segmentation.datasets.surface_types as surface_types


def evaluate(
        net,
        loader: DataLoader,
        epoch,
        criterion,
        device,
        logger=None,
        group=None,
):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    net.eval()
    loss = 0.

    with torch.no_grad():
        for i, batch in enumerate(loader):
            X = batch['X']
            Y = batch['Y']
            X = X.to(device)
            Y = Y.to(device)
            out = net.forward(X)

            if criterion:
                loss += criterion(out, Y).item()

            if i == 0:
                first_out = out
                first_batch = batch

    if criterion:
        loss /= len(loader.dataset)
        print(f'{group} loss: {loss:.4f}')

    # First epoch
    if epoch == 1 and logger:
        logger.add_images_from_path(f'{group}/images', first_batch['image_path'])
        logger.add_masks_from_path(f'{group}/masks', first_batch['mask_path'])
        logger.add_input(f'{group}/inputs', first_batch['X'].cpu())
        logger.add_target(f'{group}/targets', first_batch['Y'].cpu())

    # Every epoch
    if logger:
        logger.writer.add_scalar(f'{group}/loss', loss, epoch)
        logger.add_output(f'{group}/outputs', first_out.cpu(), epoch)
