import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from road_roughness_prediction.tools.dataset import create_surface_category_dataset
import road_roughness_prediction.models as models


def train(
        dataset,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        model_name='tiny_cnn',
        debug=False,
        cpu=False,
        save_dir=None,
):
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_data = len(dataset)
    n_validation = int(n_data * validation_split)
    indices = np.random.permutation(n_data)

    if debug:
        train_ind, validation_ind = indices[:50], indices[-10:]
    else:
        train_ind, validation_ind = indices[n_validation:], indices[:n_validation]

    print(f'total: {n_data} train: {len(train_ind)} validation: {len(validation_ind)} class: {len(dataset.categories)}')

    train_sampler = SubsetRandomSampler(train_ind)
    validation_sampler = SubsetRandomSampler(validation_ind)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

    if cpu:
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        print(f'train loss: {train_loss / batch_size:.4f} validation accuracy: {validation_accuracy:.4f}')
        if save_dir:
            torch.save(net.state_dict(), str(save_dir / f'{model_name}_dict_epoch_{i + 1:03d}.pth'))


def test(net, loader):
    total = []
    for X, labels in loader:
        outputs = net.forward(X)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total.append(c.numpy())
    return np.hstack(total).mean()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./results')
    parser.add_argument('--target-dir-name', required=True)
    parser.add_argument('--categories', nargs='+', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)

    available_networks = ['tiny_cnn', 'resnet18']
    parser.add_argument('--model-name', type=str, default='tiny_cnn', choices=available_networks)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--validation-split', type=float, default=0.2)

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
    )


if __name__ == '__main__':
    main()
