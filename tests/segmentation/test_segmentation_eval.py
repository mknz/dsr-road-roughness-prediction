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

    # Create dummy weight
    category_type = surface_types.from_string('simple')
    model_name = 'unet11'
    category_name = 'simple'
    net = models.load_model(model_name, category_type)
    weight_path = workdir / 'dummy.pth'
    torch.save(net.state_dict(), weight_path)

    data_dir = Path('tests/resources/segmentation/labelme')

    args = [
        'python3',
        'eval_seg.py',
        '--weight-path', str(weight_path),
        '--data-dirs', str(data_dir),
        '--model-name', model_name,
        '--save-path', str(workdir),
        '--category-type', category_name,
        '--seed', '1',
        '--input-size', '320', '320',
    ]

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_dummy_weight(self):
        subprocess.run(self.args, check=True)
