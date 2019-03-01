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
from road_roughness_prediction.training import evaluate


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

    def evaluate(self, fig_save_path):
        evaluate(
            net=self.net,
            loader=self.loader,
            class_names=self.loader.dataset.categories,
            fig_save_path=fig_save_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', required=True)
    parser.add_argument('--image-dir')
    parser.add_argument('--target-dir-name')
    parser.add_argument('--dir-type', choices=['deep', 'shallow'], default='shallow')
    parser.add_argument('--categories', nargs='+')
    parser.add_argument('--model-name', type=str, default='tiny_cnn')
    parser.add_argument('--fig-save-path')
    args = parser.parse_args()
    print(args)

    data_dir = Path(args.image_dir)
    assert data_dir.exists(), f'Not found {str(data_dir)}'

    fig_save_path = Path(args.fig_save_path) if args.fig_save_path else None

    evaluator = Evaluator(
        weight_path=Path(args.weight_path),
        data_dir=data_dir,
        categories=args.categories,
        model_name=args.model_name,
        target_dir_name=args.target_dir_name,
        dir_type=args.dir_type,
    )
    evaluator.evaluate(fig_save_path=fig_save_path)


if __name__ == '__main__':
    main()
