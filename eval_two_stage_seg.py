'''Evaluation script'''
import argparse
from pathlib import Path

import numpy as np

from PIL import Image

from albumentations.augmentations.functional import center_crop

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor

from road_roughness_prediction.segmentation import logging
from road_roughness_prediction.segmentation import utils
from road_roughness_prediction.segmentation.datasets import surface_types
from road_roughness_prediction.segmentation.inference import SidewalkSegmentator
import road_roughness_prediction.tools.torch as torch_tools


def _blend_image(original, segmented):
    blend = Image.blend(original.convert('RGB'), segmented.convert('RGB'), alpha=0.2)
    return blend


class ImageWriter:

    def __init__(self, save_dir: Path) -> None:
        self._counter = 0
        self.input_dir = save_dir / 'input'
        self.output_dir = save_dir / 'output'
        self.target_dir = save_dir / 'target'
        self.sidewalk_mask_dir = save_dir / 'sidewalk_mask'
        self.blend_output_dir = save_dir / 'blend_output'
        self.blend_target_dir = save_dir / 'blend_target'

        dirs = [
            self.input_dir,
            self.output_dir,
            self.target_dir,
            self.blend_output_dir,
            self.blend_target_dir,
            self.sidewalk_mask_dir,
        ]

        for dir_ in dirs:
            if not dir_.exists():
                dir_.mkdir()

    def write_images(self, inputs, segmented, sidewalk_mask, background__mask, masks=None):
        n_batches = segmented.shape[0]
        for i in range(n_batches):
            file_name = f'{self._counter:05d}'

            input_img = to_pil_image(logging.normalize(inputs[i, ::]))
            save_path = self.input_dir / (file_name + '.jpg')
            input_img.save(save_path)

            out_seg_img = segmented[i, ::]
            out_seg_index_img = utils.create_index_image(out_seg_img)
            save_path = self.output_dir / (file_name + '.png')
            out_seg_index_img.save(save_path)

            blend_output_img = _blend_image(input_img, out_seg_index_img)
            save_path = self.blend_output_dir / (file_name + '.jpg')
            blend_output_img.save(save_path)

            img_ = (sidewalk_mask[i, ::] * 255).astype(np.uint8)
            img_ = Image.fromarray(img_)
            img_ = _blend_image(input_img, img_)
            save_path = self.sidewalk_mask_dir / (file_name + '.png')
            img_.save(save_path)

            if masks is not None:
                target_img = masks[i, ::]
                target_index_img = utils.create_index_image(target_img)
                save_path = self.target_dir / (file_name + '.png')
                target_index_img.save(save_path)

                blend_target_img = _blend_image(input_img, target_index_img)
                save_path = self.blend_target_dir / (file_name + '.jpg')
                blend_target_img.save(save_path)

            self._counter += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sidewalk-detector-weight-path', required=True)
    parser.add_argument('--surface-segmentator-weight-path', required=True)

    parser.add_argument('--image-dirs', type=str, nargs='+')
    parser.add_argument('--mask-dirs', type=str, nargs='+')
    parser.add_argument('--image-paths', type=str, nargs='+')

    parser.add_argument('--save-path', default='forward')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--input-size', type=int, nargs=2, default=(640, 640))

    args = parser.parse_args()
    print(args)

    device = torch_tools.get_device(args.cpu, args.device_id)
    torch_tools.set_seeds(args.seed, device)

    sidewalk_detector_weight_path = Path(args.sidewalk_detector_weight_path)
    surface_segmentator_weight_path = Path(args.surface_segmentator_weight_path)

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    width, height = args.input_size

    segmentator = SidewalkSegmentator(
        sidewalk_detector_weight_path=sidewalk_detector_weight_path,
        surface_segmentator_weight_path=surface_segmentator_weight_path,
        device=device,
    )

    category_type = surface_types.from_string('simple')

    def _load_images(path):
        image = np.array(Image.open(path))
        image = center_crop(image, height, width)
        image = torch_tools.imagenet_normalize(image)
        return to_tensor(image).squeeze().unsqueeze(0)

    def _load_mask(path):
        mask = np.array(Image.open(path))
        mask = center_crop(mask, height, width)
        mask = surface_types.convert_mask(mask, category_type)
        return mask

    if args.image_paths:
        image_paths = [Path(path) for path in args.image_paths]
        images = [_load_images(path) for path in image_paths]
        images = torch.cat(images)
        segmented, sidewalk_mask, background_mask = segmentator.run(images)
        writer = ImageWriter(save_path)
        writer.write_images(images, segmented, sidewalk_mask, background_mask)
    else:
        image_paths = []
        mask_paths = []
        for image_dir, mask_dir in zip(args.image_dirs, args.mask_dirs):
            image_paths.extend(sorted(list(Path(image_dir).glob('*jpg'))))
            mask_paths.extend(sorted(list(Path(mask_dir).glob('*png'))))
        assert len(image_paths) == len(mask_paths)

        images = [_load_images(path) for path in image_paths]
        masks = [_load_mask(path) for path in mask_paths]

        images = torch.cat(images)
        masks = np.array(masks)

        segmented, sidewalk_mask, background_mask = segmentator.run(images)
        writer = ImageWriter(save_path)
        writer.write_images(images, segmented, sidewalk_mask, background_mask, masks)


if __name__ == '__main__':
    main()
