'''Segmentation utils'''
from pathlib import Path

import numpy as np
from PIL import Image

from road_roughness_prediction.segmentation.datasets import surface_types


def save_index_image(img: np.array, save_path: Path):
    assert len(img.shape) == 2
    assert img.dtype == np.uint8
    assert save_path.suffix == '.png'

    pil_img = Image.fromarray(img, mode='P')
    pil_img.putpalette(surface_types.COLOR_PALETTE)
    pil_img.save(save_path)
