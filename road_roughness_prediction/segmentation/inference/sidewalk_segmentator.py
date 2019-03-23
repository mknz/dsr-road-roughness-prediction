'''Two-stage segmentation'''
from pathlib import Path
from typing import Callable

import torch

import numpy as np

from road_roughness_prediction.segmentation import models
from road_roughness_prediction.segmentation.datasets import surface_types


_MODEL_NAME = 'unet11'


class SidewalkSegmentator:

    def __init__(
            self,
            sidewalk_detector_weight_path: Path,
            surface_segmentator_weight_path: Path,
            image_prep_func: Callable,
            device: torch.device,
            thr_sidewalk=0.5,
            thr_background=0.1,
    ) -> None:
        self.device = device
        self.thr_sidewalk = thr_sidewalk
        self.thr_background = thr_background

        self._sidewalk_detector = BinarySegmentator(
            sidewalk_detector_weight_path,
            image_prep_func,
            self.device,
        )

        self._surface_segmentator = MultiClassSegmentator(
            surface_segmentator_weight_path,
            image_prep_func,
            self.device,
        )

    def run(self, images: np.array) -> np.array:
        mask = self._sidewalk_detector.run(images)
        surface = self._surface_segmentator.run(images)
        seg = self._segmentate(mask, surface)
        return seg

    def _segmentate(self, mask: np.array, surface: np.array) -> np.array:
        # NOTE: This assumes channel index 0 is background
        n_batch, _, height, width = mask.shape
        out_shape = (n_batch, height, width)

        sidewalk_mask = np.full(out_shape, False)
        sidewalk_mask[mask[:, 0, ::] >= self.thr_sidewalk]  = True

        background_prob = surface[:, 0, ::]
        background_mask = np.full(out_shape, False)
        background_mask[background_prob >= self.thr_background] = True

        # Segmentation excluding background
        segmented = surface[:, 1:, ::].argmax(axis=1) + 1

        # Outside sidewalk pixels are background
        segmented[sidewalk_mask] = 0

        # Under threshold pixels are background
        segmented[background_mask] = 0

        return segmented


class BinarySegmentator:

    def __init__(self, weight_path: Path, image_prep_func: Callable, device) -> None:
        self.device = device
        self.image_prep_func = image_prep_func

        self.net = models.load_model(
            _MODEL_NAME,
            surface_types.from_string('binary')
        ).to(self.device)

        print(f'Loading {str(weight_path)}')
        state_dict = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(state_dict=state_dict)
        print('Done')

    def run(self, images: np.array) -> np.array:
        image_list = []
        for i in range(images.shape[0]):
            image_list.append(self.image_prep_func(images[i, ::]))
        X = torch.cat(image_list).squeeze().unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            out = self.net.forward(X)
        return out.cpu().numpy()


class MultiClassSegmentator:

    def __init__(self, weight_path: Path, image_prep_func: Callable, device) -> None:
        self.device = device
        self.image_prep_func = image_prep_func

        self.net = models.load_model(
            _MODEL_NAME,
            surface_types.from_string('simple')
        ).to(self.device)

        print(f'Loading {str(weight_path)}')
        state_dict = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(state_dict=state_dict)
        print('Done')

    def run(self, images: np.array) -> np.array:
        image_list = []
        for i in range(images.shape[0]):
            image_list.append(self.image_prep_func(images[i, ::]))
        X = torch.cat(image_list).squeeze().unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            out = self.net.forward(X)
        return out.cpu().numpy()
