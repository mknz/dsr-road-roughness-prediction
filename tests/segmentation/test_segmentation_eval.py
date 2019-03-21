'''Test segmentation evaluation'''
from pathlib import Path
import subprocess
import shutil
import tempfile

import torch

from road_roughness_prediction.segmentation.datasets import surface_types
from road_roughness_prediction.segmentation import models


class TestSegmentationEvaluation:

    workdir = Path(tempfile.mkdtemp())

    data_dir = Path('tests/resources/segmentation/labelme')
    image_dir = data_dir / 'JPEGImages'
    mask_dir = data_dir / 'SegmentationClassPNG'
    model_name = 'unet11'

    args = [
        'python3',
        'eval_seg.py',
        '--image-dirs', str(image_dir),
        '--mask-dirs', str(mask_dir),
        '--model-name', model_name,
        '--save-path', str(workdir),
        '--seed', '1',
        '--input-size', '320', '320',
    ]
    def _create_dummy_weight(self, category_name):
        # Create dummy weight
        category_type = surface_types.from_string(category_name)
        net = models.load_model(self.model_name, category_type)
        weight_path = self.workdir / 'dummy.pth'
        torch.save(net.state_dict(), weight_path)
        return weight_path

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def _run(self, category_name):
        '''Create dummy weight for a category and run eval script'''
        weight_path = self._create_dummy_weight(category_name)
        args_ = self.args + ['--category-type', category_name]
        args_ += ['--weight-path', str(weight_path)]
        subprocess.run(args_, check=True)

    def test_simple(self):
        self._run('simple')

    def test_binary(self):
        self._run('binary')
