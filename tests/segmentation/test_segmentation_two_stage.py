'''Test two-stage segmentation'''
from pathlib import Path
import shutil
import tempfile

import torch
import numpy as np
from PIL import Image
from albumentations.augmentations.functional import center_crop
from torchvision.transforms.functional import to_tensor

from road_roughness_prediction.segmentation.datasets import surface_types
from road_roughness_prediction.segmentation import models
from road_roughness_prediction.segmentation.inference import SidewalkSegmentator
from road_roughness_prediction.tools.torch import imagenet_normalize
from road_roughness_prediction.tools.torch import get_device


def _create_dummy_weight(category_name, model_name, workdir):
    # Create dummy weight
    category_type = surface_types.from_string(category_name)
    net = models.load_model(model_name, category_type)
    weight_path = workdir / f'{model_name}_{category_name}_dummy.pth'
    torch.save(net.state_dict(), weight_path)
    return weight_path


class TestSegmentationEvaluation:

    workdir = Path(tempfile.mkdtemp())
    input_size = (640, 640)

    data_dir = Path('tests/resources/segmentation/labelme')
    image_dir = data_dir / 'JPEGImages'
    mask_dir = data_dir / 'SegmentationClassPNG'
    model_name = 'unet11'

    sidewalk_detector_weight_path = _create_dummy_weight('binary', model_name, workdir)
    surface_segmentator_weight_path = _create_dummy_weight('simple', model_name, workdir)

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_run(self):
        images = [np.array(Image.open(p)) for p in self.image_dir.glob('*.jpg')]
        images = np.array(images)

        def _prep_func(image: np.array):
            image_ = center_crop(image, self.input_size[1], self.input_size[0])
            image_ = imagenet_normalize(image_)
            return to_tensor(image_)

        segmentator = SidewalkSegmentator(
            sidewalk_detector_weight_path=self.sidewalk_detector_weight_path,
            surface_segmentator_weight_path=self.surface_segmentator_weight_path,
            image_prep_func=_prep_func,
            device=get_device(use_cpu=True)
        )
        seg = segmentator.run(images)
        n_batch, _, height, width = images.shape
        assert seg.shape == (n_batch, self.input_size[1], self.input_size[0])
