'''Evaluate module'''
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from albumentations.augmentations.functional import center_crop

from road_roughness_prediction.segmentation import models
from road_roughness_prediction.tools.torch import make_resized_grid
from road_roughness_prediction.tools.torch import get_segmentated_image_tensor
from road_roughness_prediction.tools.torch import get_segmentated_images_tensor
import road_roughness_prediction.segmentation.datasets.surface_types as surface_types


def _load_images(paths, size=256):
    imgs = [cv2.imread(str(path)) for path in paths]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [center_crop(img, size, size) for img in imgs]
    return np.array(imgs)


def evaluate(
        net,
        loader: DataLoader,
        epoch,
        criterion,
        device,
        writer,
        group,
        jaccard_weight=0.2
):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    net.eval()
    loss = 0.

    with torch.no_grad():
        for i, batch in enumerate(loader):
            X = batch['X']
            Y = batch['Y']
            X.to(device)
            Y.to(device)
            out = net.forward(X)

            if criterion:
                loss += criterion(out, Y)

            if i == 0:
                first_out = out
                first_batch = batch

    if criterion:
        loss /= len(loader.dataset)
        print(f'{group} loss: {loss:.4f}')

    n_save = 16
    size = 256
    category_type = loader.dataset.category_type
    n_class = len(category_type)

    # First epoch
    if epoch == 1 and writer:
        image_paths = first_batch['image_path'][:n_save]
        mask_paths = first_batch['mask_path'][:n_save]

        images = _load_images(image_paths)
        masks = _load_images(mask_paths)

        writer.add_images(f'{group}/images', images/255, epoch, dataformats='NHWC')
        writer.add_images(f'{group}/masks', masks/255, epoch, dataformats='NHWC')

        # X: [n_batch, 3, height, width]
        X_ = first_batch['X'][:n_save, :, :, :]

        x_save = make_resized_grid(X_, size=size, normalize=True)
        writer.add_image(f'{group}/inputs', x_save, epoch)

        Y_ = first_batch['Y'][:n_save, :, :]
        if category_type == surface_types.BinaryCategory:
            # Y: [n_batch, 1, height, width]
            # Y is 0 or 1
            y_save = make_resized_grid(Y_, size=size, normalize=True)
        else:
            # Y: [n_batch, height, width]
            # Y is 0 to n_class - 1
            images = []
            for i in range(Y_.shape[0]):
                Y_norm = Y_[i, :, :].float() / (n_class - 1)   # Normalize 0 to 1
                images.append(get_segmentated_image_tensor(Y_norm))

            y_save = make_resized_grid(torch.stack(images), size=size, normalize=False)

        writer.add_image(f'{group}/targets', y_save, epoch)

    # Every epoch
    if writer:
        writer.add_scalar(f'{group}/loss', loss, epoch)
        if category_type == surface_types.BinaryCategory:
            # out:  [n_batch, height, width]
            out_ = first_out[:n_save, :, :]
            out_save = make_resized_grid(out_, size=size, normalize=True)
        else:
            # out:  [n_batch, n_class, height, width]
            out_ = first_out[:n_save, :, :, :]
            segmented = get_segmentated_images_tensor(out_, dim=0)
            out_save = make_resized_grid(segmented, size=size, normalize=False)

        writer.add_image(f'{group}/outputs', out_save, epoch)
