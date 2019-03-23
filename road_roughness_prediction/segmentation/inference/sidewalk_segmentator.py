'''Two-stage segmentation'''
from pathlib import Path

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
            device: torch.device,
            thr_sidewalk=0.5,
            thr_min_background=-0.2,
    ) -> None:
        self.device = device
        self.thr_sidewalk = thr_sidewalk
        self.thr_min_background = thr_min_background

        self._sidewalk_detector = BinarySegmentator(
            sidewalk_detector_weight_path,
            self.device,
        )

        self._surface_segmentator = MultiClassSegmentator(
            surface_segmentator_weight_path,
            self.device,
        )

    def run(self, images: torch.Tensor) -> np.array:
        X = images.to(self.device)
        segmented_list = []
        sidewalk_mask_list = []
        background_mask_list = []
        for i in range(X.shape[0]):
            X_ = X[i, ::].unsqueeze(0)
            mask = self._sidewalk_detector.run(X_)
            surface = self._surface_segmentator.run(X_)
            segmented, sidewalk_mask, background_mask = self._segmentate(mask, surface)
            segmented_list.append(segmented.squeeze())
            sidewalk_mask_list.append(sidewalk_mask.squeeze())
            background_mask_list.append(background_mask.squeeze())
        return np.array(segmented_list), np.array(sidewalk_mask_list), np.array(background_mask_list)

    def _segmentate(self, mask: np.array, surface: np.array) -> np.array:
        # NOTE: This assumes channel index 0 is background
        n_batch, _, height, width = mask.shape
        out_shape = (n_batch, height, width)

        sidewalk_mask = np.full(out_shape, False)
        sidewalk_mask[mask[:, 0, ::] >= self.thr_sidewalk]  = True

        # Pixels are considered to be sidewalk only if background prob < thr_min_background
        background_prob = surface[:, 0, ::]
        background_mask = np.full(out_shape, True)
        background_mask[background_prob < self.thr_min_background] = False

        # Segmentation excluding background
        segmented = surface[:, 1:, ::].argmax(axis=1) + 1

        # Outside sidewalk pixels are background
        segmented[~sidewalk_mask] = 0

        # Under threshold pixels are background
        segmented[background_mask] = 0

        return segmented.astype(np.uint8), sidewalk_mask, background_mask


class BinarySegmentator:

    def __init__(self, weight_path: Path, device) -> None:
        self.device = device

        self.net = models.load_model(
            _MODEL_NAME,
            surface_types.from_string('binary')
        ).to(self.device)

        print(f'Loading {str(weight_path)}')
        state_dict = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(state_dict=state_dict)
        print('Done')

    def run(self, images: torch.Tensor) -> np.array:
        self.net.eval()
        with torch.no_grad():
            out = self.net.forward(images)
        return out.cpu().numpy()


class MultiClassSegmentator:

    def __init__(self, weight_path: Path, device) -> None:
        self.device = device

        self.net = models.load_model(
            _MODEL_NAME,
            surface_types.from_string('simple')
        ).to(self.device)

        print(f'Loading {str(weight_path)}')
        state_dict = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(state_dict=state_dict)
        print('Done')

    def run(self, images: torch.Tensor) -> np.array:
        self.net.eval()
        with torch.no_grad():
            out = self.net.forward(images)
        return out.cpu().numpy()
