'''Segmentation training'''
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader

from albumentations import Compose
from albumentations import RandomCrop
from albumentations import CenterCrop
from albumentations import HorizontalFlip
from albumentations import Normalize

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from road_roughness_prediction.segmentation.datasets import SidewalkSegmentationDatasetFactory
import road_roughness_prediction.segmentation.datasets.surface_types as surface_types
from road_roughness_prediction.segmentation import models
from road_roughness_prediction.segmentation.inference import evaluate
from road_roughness_prediction.tools.torch import make_resized_grid
from road_roughness_prediction.tools.torch import get_segmentated_images_tensor


def train(net, loader, epoch, optimizer, criterion, device, writer, model_name):
    total_loss = 0.
    net.train()
    for i, batch in enumerate(tqdm(loader)):
        X = batch['X']
        Y = batch['Y']
        X.to(device)
        Y.to(device)

        optimizer.zero_grad()
        out = net.forward(X)

        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i == 0:
            first_batch = batch
            first_out = out

    total_loss /= len(loader.dataset)
    print(f'train loss: {total_loss:.4f}')

    # Record loss
    writer.add_scalar('train/loss', total_loss, epoch)

    n_save = 16
    size = 256
    category_type = loader.dataset.category_type
    if category_type == surface_types.BinaryCategory:
        save_img = make_resized_grid(first_out[:n_save, :, :], size=size, normalize=True)
    else:
        segmented = get_segmentated_images_tensor(first_out[:n_save, :, :, :], dim=0)
        save_img = make_resized_grid(segmented, size=size, normalize=False)
    writer.add_image('train/outputs', save_img, epoch)

    # Save model
    save_path = Path(writer.log_dir) / f'{model_name}_dict_epoch_{epoch:03d}.pth'
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-dirs', required=True, nargs='+')
    parser.add_argument('--validation-data-dirs', required=True, nargs='+')
    parser.add_argument('--save-dir', default='./runs')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--category-type', default='binary', choices=['binary', 'simple'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input-size', type=int, nargs=2, default=(640, 640))
    parser.add_argument('--jaccard-weight', type=float, default=0.3)

    available_networks = ['unet11']
    parser.add_argument('--model-name', type=str, default='unet11', choices=available_networks)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--run-name', type=str, required=True)

    args = parser.parse_args()
    print(args)

    # Setting rand seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.cpu:
        device = 'cpu'
    else:
        device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else "cpu")
        torch.cuda.manual_seed(seed)

    train_data_dirs = [Path(p) for p in args.train_data_dirs]
    for data_dir in train_data_dirs:
        assert data_dir.exists(), f'{str(data_dir)} does not exist.'

    validation_data_dirs = [Path(p) for p in args.validation_data_dirs]
    for data_dir in validation_data_dirs:
        assert data_dir.exists(), f'{str(data_dir)} does not exist.'

    log_dir = _get_log_dir(args)
    writer = SummaryWriter(str(log_dir))
    writer.add_text('args', str(args))

    input_size = args.input_size

    # Transforms
    train_transform = Compose([
        HorizontalFlip(p=0.5),
        RandomCrop(*input_size),
    ])

    validation_transform = Compose([
        CenterCrop(*input_size),
    ])

    if args.category_type == 'binary':
        category_type = surface_types.BinaryCategory
    elif args.category_type == 'simple':
        category_type = surface_types.SimpleCategory

    # Train dataset and loader
    train_dataset = SidewalkSegmentationDatasetFactory(
        train_data_dirs,
        category_type,
        train_transform,
    )
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    # Validation dataset and loader
    validation_dataset = SidewalkSegmentationDatasetFactory(
        validation_data_dirs,
        category_type,
        validation_transform,
    )
    validation_loader = DataLoader(validation_dataset, args.batch_size, shuffle=False)

    model_name = args.model_name
    if model_name == 'unet11':
        if category_type == surface_types.BinaryCategory:
            net = models.UNet11(pretrained=True)
        else:
            net = models.UNet11(num_classes=len(category_type), pretrained=True)
    else:
        raise ValueError(model_name)

    jaccard_weight = args.jaccard_weight
    if category_type == surface_types.BinaryCategory:
        criterion = models.loss.LossBinary(jaccard_weight)
    else:
        criterion = models.loss.LossMulti(jaccard_weight, num_classes=len(category_type))

    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(1, args.epochs + 1):
        print(f'epoch: {epoch:03d}')
        sys.stdout.flush()
        train(net, train_loader, epoch, optimizer, criterion, device, writer, model_name)
        evaluate(net, validation_loader, epoch, criterion, device, writer, 'validation', jaccard_weight)


if __name__ == '__main__':
    main()
