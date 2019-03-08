'''Evaluate module'''
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from road_roughness_prediction.segmentation import models


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_image(image, width=256):
    h, w, _ = image.shape
    rate = width / w
    return cv2.resize(image, (int(w * rate), int(h * rate)))


def evaluate(net, loader: DataLoader, epoch=None, writer=None, group=None, params={}):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    jaccard_weight = params['jaccard_weight']

    net.eval()
    loss = 0.

    outputs = []
    n_save = 8

    criterion = models.LossBinary(jaccard_weight)
    with torch.no_grad():
        for batch in loader:
            X = batch['X']
            Y = batch['Y']
            out = net.forward(X)
            loss += criterion(out, Y)

            outputs.append(out)
            if epoch == 1 and writer:  # Write raw images only once
                image_paths = batch['image_path']
                mask_paths = batch['mask_path']
                if len(image_paths) > n_save:
                    image_paths_ = image_paths[:n_save]
                    mask_paths_ = mask_paths[:n_save]
                    X_ = X[:n_save, :, :, :]
                    Y_ = Y[:n_save, :, :]
                else:
                    image_paths_ = image_paths
                    mask_paths_ = mask_paths
                    X_ = X
                    Y_ = Y

                images = np.array([_resize_image(load_image(p)) for p in image_paths_])
                masks = np.array([_resize_image(load_image(p)) for p in mask_paths_])

                writer.add_images(f'{group}/images', images/255, epoch, dataformats='NHWC')
                writer.add_images(f'{group}/masks', masks/255, epoch, dataformats='NHWC')

                x_save = make_grid(X_, normalize=True)
                writer.add_image(f'{group}/inputs', x_save, epoch)

                y_save = make_grid(Y_, normalize=True)
                writer.add_image(f'{group}/targets', y_save, epoch)

    loss /= len(loader.dataset)
    print(f'---{group}---')
    print(f'loss: {loss:.4f}')

    # TensorboardX
    if writer:
        writer.add_scalar(f'{group}/loss', loss, epoch)

        outputs = torch.cat(outputs, dim=0)
        if outputs.shape[0] > n_save:
            outputs_ = outputs[:n_save, :, :, :]
        else:
            outputs_ = outputs

        out_save = make_grid(outputs_, normalize=True)
        writer.add_image(f'{group}/outputs', out_save, epoch)
