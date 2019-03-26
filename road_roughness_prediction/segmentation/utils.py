'''Segmentation utils'''
import io
from pathlib import Path

import numpy as np
from PIL import Image

from road_roughness_prediction.segmentation.datasets import surface_types


def create_index_image(img: np.array):
    assert len(img.shape) == 2
    assert img.dtype == np.uint8

    pil_img = Image.fromarray(img, mode='P')
    pil_img.putpalette(surface_types.COLOR_PALETTE)
    return pil_img


def pil_image_to_bytes(image: Image, format):
    with io.BytesIO() as buf:
        image.save(buf, format=format)
        buf.seek(0)
        bytes_ = buf.read()
    return bytes_
