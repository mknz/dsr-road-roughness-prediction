import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from tensorboardX import SummaryWriter

from road_roughness_prediction.segmentation.datasets import SidewalkSegmentationDatasetFactory
from road_roughness_prediction.segmentation.datasets.surface_types import SimpleCategory
from road_roughness_prediction.segmentation import models
from road_roughness_prediction.segmentation.inference import evaluate


def _get_weights(dataset, validation_split):
    '''Create weights for sampler'''

    # Init weights with zero
    weights_train = [0. for _ in range(len(dataset))]
    weights_validation = [0. for _ in range(len(dataset))]

    for i in range(len(dataset)):
        weight = 1.
        if np.random.random() > validation_split:
            weights_train[i] = weight
        else:
            weights_validation[i] = weight

    return weights_train, weights_validation


def _get_loader(
        dataset,
        batch_size,
        validation_split=0.2,
):
    n_data = len(dataset)
    weights_train, weights_validation = _get_weights(dataset, validation_split)

    train_sampler = WeightedRandomSampler(weights_train, num_samples=n_data - 1)
    validation_sampler = WeightedRandomSampler(weights_validation, num_samples=n_data - 1)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
    return train_loader, validation_loader


def train(
        dataset,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        model_name='unet11',
        debug=False,
        cpu=False,
        log_dir=None,
        seed=1,
        device_id=0,
):
    seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(str(log_dir))

    train_loader, validation_loader = _get_loader(
        dataset,
        batch_size,
        validation_split,
    )

    if cpu:
        device = 'cpu'
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else "cpu")
        torch.cuda.manual_seed(seed)

    model_name = 'unet11'
    net = models.UNet11(pretrained=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch:03d}')
        sys.stdout.flush()
        train_loss = 0.
        net.train()
        for X, Y in tqdm(train_loader):
            X.to(device)
            Y.to(device)

            optimizer.zero_grad()
            outputs = net.forward(X)

            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        print('---train---')
        print(f'loss: {train_loss:.4f}')
        writer.add_scalar('train/loss', train_loss, epoch)

        evaluate(
            net=net,
            loader=validation_loader,
            epoch=epoch,
            writer=writer,
            group='validation',
        )

        torch.save(net.state_dict(), str(log_dir / f'{model_name}_dict_epoch_{epoch:03d}.pth'))


def _get_log_dir(args) -> Path:
    '''Construct log directory path'''
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = Path(args.save_dir)
    log_dir = root / args.model_name / args.transform / args.run_id / current_time
    if not log_dir.parent.exists():
        log_dir.parent.mkdir(parents=True)
    return log_dir


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', required=True, nargs='+')
    parser.add_argument('--save-dir', default='./runs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--class-balanced', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)

    available_networks = ['unet11']
    parser.add_argument('--model-name', type=str, default='unet11', choices=available_networks)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--transform', default='basic_transform')
    parser.add_argument('--run-id', default='standard')

    args = parser.parse_args()
    print(args)

    data_dirs = [Path(p) for p in args.data_dirs]
    for data_dir in data_dirs:
        assert data_dir.exists(), f'{str(data_dir)} does not exist.'

    log_dir = _get_log_dir(args)

    from albumentations import Compose
    from albumentations import RandomCrop

    transform = Compose([
        RandomCrop(256, 256),
    ])

    is_binary = True
    category_type = SimpleCategory
    dataset = SidewalkSegmentationDatasetFactory(
        data_dirs,
        category_type,
        transform,
        is_binary,
    )

    train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_name=args.model_name,
        debug=args.debug,
        cpu=args.cpu,
        log_dir=log_dir,
        seed=args.seed,
        device_id=args.device_id,
    )


if __name__ == '__main__':
    main()
