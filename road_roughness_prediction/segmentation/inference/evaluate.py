'''Evaluate module'''
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from albumentations.augmentations.functional import center_crop

from road_roughness_prediction.segmentation import models
from road_roughness_prediction.tools.torch import make_resized_grid


def _load_images(paths, size=256):
    imgs = [cv2.imread(str(path)) for path in paths]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [center_crop(img, size, size) for img in imgs]
    return np.array(imgs)


def evaluate(
        net,
        loader: DataLoader,
        epoch=None,
        device=None,
        writer=None,
        group=None,
        jaccard_weight=0.2
):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    net.eval()
    loss = 0.

    criterion = models.LossBinary(jaccard_weight)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            X = batch['X']
            Y = batch['Y']
            X.to(device)
            Y.to(device)

            out = net.forward(X)
            loss += criterion(out, Y)

            if i == 0:
                first_out = out
                first_batch = batch

    loss /= len(loader.dataset)
    print(f'{group} loss: {loss:.4f}')

    n_save = 16
    size = 256

    # First epoch
    if epoch == 1 and writer:
        image_paths = first_batch['image_path'][:n_save]
        mask_paths = first_batch['mask_path'][:n_save]
        X_ = first_batch['X'][:n_save, :, :, :]
        Y_ = first_batch['Y'][:n_save, :, :]

        images = _load_images(image_paths)
        masks = _load_images(mask_paths)

        writer.add_images(f'{group}/images', images/255, epoch, dataformats='NHWC')
        writer.add_images(f'{group}/masks', masks/255, epoch, dataformats='NHWC')

        x_save = make_resized_grid(X_, size=size, normalize=True)
        writer.add_image(f'{group}/inputs', x_save, epoch)

        y_save = make_resized_grid(Y_, size=size, normalize=True)
        writer.add_image(f'{group}/targets', y_save, epoch)

    # Every epoch
    if writer:
        writer.add_scalar(f'{group}/loss', loss, epoch)
        out_ = first_out[:n_save, :, :, :]
        out_save = make_resized_grid(out_, size=size, normalize=True)
        writer.add_image(f'{group}/outputs', out_save, epoch)
