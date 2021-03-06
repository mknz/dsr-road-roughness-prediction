'''Segmentation training'''
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from albumentations import Compose
from albumentations import OneOf
from albumentations import RandomCrop
from albumentations import CenterCrop
from albumentations import HorizontalFlip
from albumentations import RandomCrop
from albumentations import RandomScale
from albumentations import RandomGamma
from albumentations import Resize
from albumentations import Rotate
from albumentations import RandomSizedCrop
from albumentations import HueSaturationValue
from albumentations import RandomBrightnessContrast
from albumentations.imgaug.transforms import IAAPerspective


from road_roughness_prediction.segmentation import datasets
from road_roughness_prediction.segmentation import models
from road_roughness_prediction.segmentation.inference import evaluate
from road_roughness_prediction.segmentation import logging
import road_roughness_prediction.tools.torch as torch_tools


def train(net, loader, epoch, optimizer, criterion, device, logger):
    losses = []
    net.train()
    for i, batch in enumerate(tqdm(loader)):
        X = batch['X']
        Y = batch['Y']
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        out = net.forward(X)

        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if i == 0:
            first_out = out

    mean_loss = np.mean(losses)
    print(f'train loss: {mean_loss:.4f}')

    # Record loss
    logger.writer.add_scalar('train/loss', mean_loss, epoch)

    # Record output
    logger.add_output('train/outputs', first_out.cpu(), epoch)

    # Save model
    model_name = str(net).split('(')[0].lower()
    save_path = Path(logger.writer.logdir) / f'{model_name}_dict_epoch_{epoch:03d}.pth'
    torch.save(net.state_dict(), str(save_path))


def _get_log_dir(args) -> Path:
    '''Construct log directory path'''
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = Path(args.save_dir)
    log_dir = root / args.model_name / args.run_name / current_time
    if not log_dir.parent.exists():
        log_dir.parent.mkdir(parents=True)
    return log_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-image-dirs', required=True, nargs='+')
    parser.add_argument('--train-mask-dirs', required=True, nargs='+')
    parser.add_argument('--train-dataset-types', nargs='+')

    parser.add_argument('--validation-image-dirs', required=True, nargs='+')
    parser.add_argument('--validation-mask-dirs', required=True, nargs='+')
    parser.add_argument('--validation-dataset-types', nargs='+')

    parser.add_argument('--save-dir', default='./runs')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--category-type', default='binary', choices=['binary', 'simple'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input-size', type=int, nargs=2, default=(640, 640))
    parser.add_argument('--jaccard-weight', type=float, default=0.3)

    available_networks = ['unet11', 'unet16']
    parser.add_argument('--model-name', type=str, default='unet11', choices=available_networks)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--run-name', type=str, required=True)

    args = parser.parse_args()
    print(args)

    device = torch_tools.get_device(args.cpu, args.device_id)
    torch_tools.set_seeds(args.seed, device)

    train_image_dirs = [Path(p) for p in args.train_image_dirs]
    train_mask_dirs = [Path(p) for p in args.train_mask_dirs]
    validation_image_dirs = [Path(p) for p in args.validation_image_dirs]
    validation_mask_dirs = [Path(p) for p in args.validation_mask_dirs]

    for data_dir in\
            train_image_dirs +\
            train_mask_dirs +\
            validation_image_dirs +\
            validation_mask_dirs:
        assert data_dir.exists(), f'{str(data_dir)} does not exist.'

    input_size = args.input_size
    w, h = input_size
    rate = 0.9

    '''
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        RandomCrop(*input_size),
    ])
    '''
    # Transforms
    transforms = {}

    # Basic transform
    transforms['base'] = Compose([
        HorizontalFlip(p=0.5),
        #IAAPerspective(scale=(0.05, 0.1), p=0.3),
        Rotate(5, p=0.5),
        RandomGamma(p=0.5),
        HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.5
        ),
        RandomBrightnessContrast(p=0.5),
        OneOf([
            RandomSizedCrop((int(h * rate), int(w * rate)), h, w, p=1.0),
            RandomCrop(h, w, p=1.0),
        ], p=1.0)
    ])

    # BDD dataset
    transforms['bdd'] = transforms['base']

    # Always shrink to 22% - 50%
    transforms['walk'] = Compose([
        HorizontalFlip(p=0.5),
        Rotate(5, p=0.5),
        RandomGamma(p=0.5),
        HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.5
        ),
        RandomBrightnessContrast(p=0.5),
        RandomScale((-0.78, -0.5), p=1.0),
        RandomCrop(h, w, p=1.0),
    ])

    # MISC dataset transform
    transforms['misc'] = Compose([
        HorizontalFlip(p=0.5),
        Rotate(5, p=0.5),
        RandomGamma(p=0.5),
        HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.5
        ),
        RandomBrightnessContrast(p=0.5),
        Resize(h, w, p=1.0),
    ])

    validation_transform = Compose([
        CenterCrop(h, w),
    ])

    category_type = datasets.surface_types.from_string(args.category_type)

    # Logger
    log_dir = _get_log_dir(args)
    logger = logging.Logger(log_dir, n_save=16, image_size=256, category_type=category_type)
    logger.writer.add_text('args', str(args))

    train_datasets = []
    for image_dir, mask_dir, dataset_type in zip(train_image_dirs, train_mask_dirs, args.train_dataset_types):
        _dataset = datasets.create_dataset(
            dataset_type,
            [image_dir],
            [mask_dir],
            category_type,
            transforms[dataset_type],
        )
        train_datasets.append(_dataset)

    validation_datasets = []
    for image_dir, mask_dir, dataset_type in zip(validation_image_dirs, validation_mask_dirs, args.validation_dataset_types):
        _dataset = datasets.create_dataset(
            dataset_type,
            [image_dir],
            [mask_dir],
            category_type,
            validation_transform,
        )
        validation_datasets.append(_dataset)

    # Merge datasets
    train_dataset = ConcatDataset(train_datasets)
    validation_dataset = ConcatDataset(validation_datasets)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, args.batch_size, shuffle=False)
    net = models.load_model(args.model_name, category_type).to(device)

    criterion = models.loss.get_criterion(category_type, args.jaccard_weight)
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(1, args.epochs + 1):
        print(f'epoch: {epoch:03d}')
        sys.stdout.flush()
        train(net, train_loader, epoch, optimizer, criterion, device, logger)
        evaluate(net, validation_loader, epoch, criterion, device, logger, 'validation')


if __name__ == '__main__':
    main()
