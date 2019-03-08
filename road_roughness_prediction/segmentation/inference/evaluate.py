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


def evaluate(net, loader: DataLoader, epoch=None, device=None, writer=None, group=None, params={}):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    jaccard_weight = params['jaccard_weight']

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

    n_save = 8

    # First epoch
    if epoch == 1 and writer:
        image_paths = first_batch['image_path'][:n_save]
        mask_paths = first_batch['mask_path'][:n_save]
        X_ = first_batch['X'][:n_save, :, :, :]
        Y_ = first_batch['Y'][:n_save, :, :]

        images = np.array([_resize_image(load_image(p)) for p in image_paths])
        masks = np.array([_resize_image(load_image(p)) for p in mask_paths])

        writer.add_images(f'{group}/images', images/255, epoch, dataformats='NHWC')
        writer.add_images(f'{group}/masks', masks/255, epoch, dataformats='NHWC')

        x_save = make_grid(X_, normalize=True)
        writer.add_image(f'{group}/inputs', x_save, epoch)

        y_save = make_grid(Y_, normalize=True)
        writer.add_image(f'{group}/targets', y_save, epoch)

    # Every epoch
    if writer:
        writer.add_scalar(f'{group}/loss', loss, epoch)
        out_ = first_out[:n_save, :, :, :]
        out_save = make_grid(out_, normalize=True)
        writer.add_image(f'{group}/outputs', out_save, epoch)
