import argparse
from pathlib import Path
from typing import List

import numpy as np

import torch
from torch.utils.data import DataLoader

from road_roughness_prediction import models
from road_roughness_prediction.tools.dataset import create_surface_category_test_dataset


np.set_printoptions(precision=4)


def forward(weight_path: Path, data_dir, categories: List[str], model_name: str):
    img_size = 256
    n_class = len(categories)
    dataset = create_surface_category_test_dataset(
        data_dir,
        categories,
        output_size=img_size,
    )
    dataset.show_dist()
    loader = DataLoader(dataset)

    if model_name == 'tiny_cnn':
        net = models.TinyCNN(n_class)
    elif model_name == 'resnet18':
        net = models.Resnet18(n_class)

    net.load_state_dict(state_dict=torch.load(weight_path))

    test(net, loader, len(categories))


def test(net, loader: DataLoader, n_class):
    class_count = [0 for _ in range(n_class)]
    class_correct = [0 for _ in range(n_class)]
    for X, labels in loader:
        outputs = net.forward(X)
        _, predicted = torch.max(outputs, 1)
        for pred, label in zip(predicted.tolist(), labels.tolist()):
            class_count[int(label)] += 1
            class_correct[int(label)] += int(pred == label)

    accuracy = sum(class_correct) / sum(class_count)
    class_accuracy = [
        correct / count if count > 0 else 0.
        for correct, count
        in zip(class_correct, class_count)
    ]

    print(f'total_accuracy: {accuracy}')
    print(f'class_accuracy: {class_accuracy}')
    print(f'class_count: {class_count}')
    print(f'class_correct: {class_correct}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', required=True)
    parser.add_argument('--image-dir')
    parser.add_argument('--categories', nargs='+')
    parser.add_argument('--model-name', type=str, default='tiny_cnn')
    args = parser.parse_args()

    forward(Path(args.weight_path), Path(args.image_dir), args.categories, args.model_name)


if __name__ == '__main__':
    main()
