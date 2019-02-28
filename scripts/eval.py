'''Evaluation script'''
import argparse
from pathlib import Path
from typing import List

import numpy as np

import torch
from torch.utils.data import DataLoader

from road_roughness_prediction.config import EvalConfig
from road_roughness_prediction import models
from road_roughness_prediction.dataset import SurfaceCategoryDatasetFactory
from road_roughness_prediction.dataset.transformations import TransformFactory


np.set_printoptions(precision=4)


def evaluate(
        weight_path: Path,
        data_dir,
        target_dir_name,
        categories: List[str],
        model_name: str,
        dir_type: str,
):
    n_class = len(categories)

    config = EvalConfig()
    transform = TransformFactory(config)

    dataset = SurfaceCategoryDatasetFactory(
        data_dir,
        categories,
        target_dir_name,
        dir_type,
        transform,
    )
    assert len(dataset), f'No data found in {data_dir}'
    dataset.show_dist()
    loader = DataLoader(dataset, batch_size=100)

    if model_name == 'tiny_cnn':
        net = models.TinyCNN(n_class)
    elif model_name == 'resnet18':
        net = models.Resnet18(n_class)

    net.load_state_dict(state_dict=torch.load(weight_path))

    test(net, loader, len(categories))


def test(net, loader: DataLoader, n_class):
    net.eval()
    class_count = [0 for _ in range(n_class)]
    class_correct = [0 for _ in range(n_class)]
    with torch.no_grad():
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
    parser.add_argument('--target-dir-name')
    parser.add_argument('--dir-type', choices=['deep', 'shallow'], default='shallow')
    parser.add_argument('--categories', nargs='+')
    parser.add_argument('--model-name', type=str, default='tiny_cnn')
    args = parser.parse_args()

    data_dir = Path(args.image_dir)
    assert data_dir.exists(), f'Not found {str(data_dir)}'

    evaluate(
        weight_path=Path(args.weight_path),
        data_dir=data_dir,
        categories=args.categories,
        model_name=args.model_name,
        target_dir_name=args.target_dir_name,
        dir_type=args.dir_type,
    )


if __name__ == '__main__':
    main()
