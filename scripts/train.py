import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from road_roughness_prediction.tools.dataset import create_surface_category_dataset
from road_roughness_prediction import models


np.set_printoptions(precision=4)


def _get_weights(dataset, validation_split, is_class_balanced=False):
    '''Create weights for sampler, optionally class-balanced'''

    # Class distribution dict
    dist_dict = {}
    for i, _, _, dist in dataset.distributions:
        dist_dict[i] = dist

    # Init weights with zero
    weights_train = [0. for _ in range(len(dataset))]
    weights_validation = [0. for _ in range(len(dataset))]
    for i, (label, path) in enumerate(zip(dataset.labels, dataset.paths)):
        # No class examples
        if dist_dict[label] == 0.:
            continue

        if is_class_balanced:
            weight = 1. / dist_dict[label]
        else:
            weight = 1.

        if np.random.random() > validation_split:
            weights_train[i] = weight
        else:
            weights_validation[i] = weight

    return weights_train, weights_validation


def create_train_validation_split_loader(
        dataset,
        batch_size,
        validation_split=0.2,
        is_class_balanced=False,
):
    n_data = len(dataset)
    weights_train, weights_validation = _get_weights(
        dataset,
        validation_split,
        is_class_balanced=is_class_balanced,
    )

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
        model_name='tiny_cnn',
        debug=False,
        cpu=False,
        save_dir=None,
        is_class_balanced=False,
        seed=1,
        device_id=0,
):
    seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, validation_loader = create_train_validation_split_loader(
        dataset,
        batch_size,
        validation_split,
        is_class_balanced=is_class_balanced,
    )

    if cpu:
        device = 'cpu'
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else "cpu")
        torch.cuda.manual_seed(seed)

    if model_name == 'tiny_cnn':
        net = models.TinyCNN(n_out=len(dataset.categories))
    elif model_name == 'resnet18':
        net = models.Resnet18(n_out=len(dataset.categories))
    else:
        ValueError(f'Unknown model name {model_name}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for i in range(epochs):
        print(f'epoch: {i + 1:03d}')
        sys.stdout.flush()
        train_loss = 0.
        net.train()
        for X, labels in tqdm(train_loader):
            X.to(device)
            labels.to(device)

            optimizer.zero_grad()
            outputs = net.forward(X)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        net.eval()
        validation_accuracy = test(net, validation_loader)
        print(f'train loss: {train_loss / batch_size:.4f}')
        if save_dir:
            torch.save(net.state_dict(), str(save_dir / f'{model_name}_dict_epoch_{i + 1:03d}.pth'))


def test(net, loader: DataLoader):
    total = []
    for X, labels in loader:
        outputs = net.forward(X)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total.append(c.numpy())
    accuracy = np.hstack(total).mean()
    class_accuracy = np.vstack(total).mean(axis=0)
    print(f'accuracy: {accuracy}')
    print(f'class_accuracy: {class_accuracy}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./results')
    parser.add_argument('--target-dir-name', required=True)
    parser.add_argument('--categories', nargs='+', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--class-balanced', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)

    available_networks = ['tiny_cnn', 'resnet18']
    parser.add_argument('--model-name', type=str, default='tiny_cnn', choices=available_networks)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device-id', type=int, default=0)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    categories = args.categories
    target_dir_name = args.target_dir_name
    assert data_dir.exists(), f'{str(data_dir)} does not exist.'

    save_dir = Path(args.save_dir) / str(time.time())
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    dataset = create_surface_category_dataset(data_dir, categories, target_dir_name)

    # Show class distribution
    dataset.show_dist()

    train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_name=args.model_name,
        debug=args.debug,
        cpu=args.cpu,
        save_dir=save_dir,
        seed=args.seed,
        is_class_balanced=args.class_balanced,
    )


if __name__ == '__main__':
    main()
