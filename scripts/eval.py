'''Evaluation script'''
import argparse
from pathlib import Path
from typing import List

import numpy as np

import torch
from torch.utils.data import DataLoader

from road_roughness_prediction.config import Config
from road_roughness_prediction import models
from road_roughness_prediction.datasets import SurfaceCategoryDatasetFactory
from road_roughness_prediction.datasets.transformations import TransformFactory
from road_roughness_prediction.datasets.transformations import TransformType
from road_roughness_prediction.tools import calc_plot_confusion_matrix


np.set_printoptions(precision=4)


class Evaluator:
    def __init__(
            self,
            weight_path: Path,
            data_dir,
            target_dir_name,
            categories: List[str],
            model_name: str,
            dir_type: str,
    )-> None:
        self.categories = categories
        self.n_class = len(categories)

        self.config = Config()
        self.config.from_dict(dict(TRANSFORMATION=TransformType.BASIC_EVAL_TRANSFORM))
        transform = TransformFactory(self.config)

        dataset = SurfaceCategoryDatasetFactory(
            data_dir,
            categories,
            target_dir_name,
            dir_type,
            transform,
        )
        assert len(dataset), f'No data found in {data_dir}'
        dataset.show_dist()

        self.loader = DataLoader(dataset, batch_size=100)

        if model_name == 'tiny_cnn':
            self.net = models.TinyCNN(self.n_class)
        elif model_name == 'resnet18':
            self.net = models.Resnet18(self.n_class)

        self.net.load_state_dict(state_dict=torch.load(weight_path))

    def evaluate(self, save_fig_path):
        self.net.eval()

        class_count = [0 for _ in range(self.n_class)]
        class_correct = [0 for _ in range(self.n_class)]
        y_test = []
        y_pred = []

        with torch.no_grad():
            for X, labels in self.loader:
                outputs = self.net.forward(X)
                _, predicted = torch.max(outputs, 1)
                labels_ = labels.tolist()
                predicted_ = predicted.tolist()
                y_test += labels_
                y_pred += predicted_
                for label, pred in zip(labels_, predicted_):
                    class_count[int(label)] += 1
                    class_correct[int(label)] += int(pred == label)

        calc_plot_confusion_matrix(y_test, y_pred, self.categories, save_fig_path)

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
    parser.add_argument('--save-fig-path')
    args = parser.parse_args()

    data_dir = Path(args.image_dir)
    assert data_dir.exists(), f'Not found {str(data_dir)}'

    save_fig_path = Path(args.save_fig_path) if args.save_fig_path else None

    evaluator = Evaluator(
        weight_path=Path(args.weight_path),
        data_dir=data_dir,
        categories=args.categories,
        model_name=args.model_name,
        target_dir_name=args.target_dir_name,
        dir_type=args.dir_type,
    )
    evaluator.evaluate(save_fig_path=save_fig_path)


if __name__ == '__main__':
    main()
