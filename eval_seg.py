'''Evaluation script'''
import argparse
from pathlib import Path

import numpy as np

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader

from PIL import Image

from albumentations import Compose
from albumentations import CenterCrop

from torchvision.transforms.functional import to_pil_image

from road_roughness_prediction.segmentation.datasets import SidewalkSegmentationDatasetFactory
from road_roughness_prediction.segmentation.datasets import surface_types
from road_roughness_prediction.segmentation import models
from road_roughness_prediction.segmentation import logging
from road_roughness_prediction.segmentation import utils
import road_roughness_prediction.tools.torch as torch_tools


def evaluate(net, loader: DataLoader, criterion, device, save_dir):

    net.eval()
    losses = []
    image_writer = ImageWriter(save_dir)
    with torch.no_grad():
        for batch in tqdm(loader):
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            out = net.forward(X)
            losses.append(criterion(out, Y).item())
            image_writer.write_batch_images(batch, out.cpu())

    mean_loss = np.mean(losses)
    print(f'loss: {mean_loss:.4f}')


class ImageWriter:

    def __init__(self, save_dir: Path):
        self._counter = 0

        self.input_dir = save_dir / 'input'
        self.output_dir = save_dir / 'output'
        self.target_dir = save_dir / 'target'
        self.blend_output_dir = save_dir / 'blend_output'
        self.blend_target_dir = save_dir / 'blend_target'

        dirs = [
            self.input_dir,
            self.output_dir,
            self.target_dir,
            self.blend_output_dir,
            self.blend_target_dir,
        ]

        for dir_ in dirs:
            if not dir_.exists():
                dir_.mkdir()

    def write_batch_images(self, batch, out):
        '''Write batch-wise data into images'''

        X = batch['X']
        Y = batch['Y']

        # out:  [n_batch, n_class, height, width]
        out_seg = out.argmax(1)

        n_batches = X.shape[0]
        for i in range(n_batches):
            file_name = f'{self._counter:05d}'

            input_img = to_pil_image(logging.normalize(X[i, :, :, :]))
            save_path = self.input_dir / (file_name + '.jpg')
            input_img.save(save_path)

            out_seg_img = np.array(out_seg[i, :, :]).astype(np.uint8)
            out_seg_index_img = utils.create_index_image(out_seg_img)
            save_path = self.output_dir / (file_name + '.png')
            out_seg_index_img.save(save_path)

            target_img = np.array(Y[i, :, :]).astype(np.uint8)
            target_index_img = utils.create_index_image(target_img)
            save_path = self.target_dir / (file_name + '.png')
            target_index_img.save(save_path)

            blend_output_img = self._blend_image(input_img, out_seg_index_img)
            save_path = self.blend_output_dir / (file_name + '.jpg')
            blend_output_img.save(save_path)

            blend_target_img = self._blend_image(input_img, target_index_img)
            save_path = self.blend_target_dir / (file_name + '.jpg')
            blend_target_img.save(save_path)

            self._counter += 1

    def _blend_image(self, original, segmented):
        blend = Image.blend(original.convert('RGB'), segmented.convert('RGB'), alpha=0.2)
        return blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', required=True)
    parser.add_argument('--data-dirs', required=True, type=str, nargs='+')
    parser.add_argument('--model-name', type=str, default='unet11')
    parser.add_argument('--save-path', default='forward')
    parser.add_argument('--category-type', default='binary', choices=['binary', 'simple'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--input-size', type=int, nargs=2, default=(640, 640))
    parser.add_argument('--jaccard-weight', type=float, default=0.3)

    args = parser.parse_args()
    print(args)

    data_dirs = [Path(p) for p in args.data_dirs]
    for data_dir in data_dirs:
        assert data_dir.exists(), f'{str(data_dir)} does not exist.'

    device = torch_tools.get_device(args.cpu, args.device_id)
    torch_tools.set_seeds(args.seed, device)

    weight_path = Path(args.weight_path)

    category_type = surface_types.from_string(args.category_type)

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    net = models.load_model(args.model_name, category_type).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict=state_dict)

    input_size = args.input_size

    transform = Compose([
        CenterCrop(*input_size),
    ])

    dataset = SidewalkSegmentationDatasetFactory(
        data_dirs,
        category_type,
        transform,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = models.loss.get_criterion(category_type, args.jaccard_weight)
    evaluate(net, loader, criterion, device, save_path)


if __name__ == '__main__':
    main()
