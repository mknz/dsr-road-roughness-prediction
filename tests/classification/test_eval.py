'''Test evaluation code'''
from pathlib import Path
import subprocess
import shutil
import tempfile

import pytest
import torch

from road_roughness_prediction.classification.config import Config
from road_roughness_prediction.classification.datasets.transformations import TransformType
from road_roughness_prediction.classification.datasets.surface_types import SurfaceBasicCategory
from road_roughness_prediction.classification import models


class TestEvaluation:

    config = Config()
    config.from_dict(dict(TRANSFORMATION=TransformType.BASIC_EVAL_TRANSFORM))

    workdir = Path(tempfile.mkdtemp())
    n_class = len(SurfaceBasicCategory)

    # Create dummy weight
    net = models.TinyCNN(n_class)
    weight_path = workdir / 'dummy.pth'
    torch.save(net.state_dict(), weight_path)

    image_dir = Path('tests/resources/surfaces')

    args = [
        'python3',
        'scripts/classification/eval.py',
        '--weight-path', str(weight_path),
        '--image-dir', str(image_dir),
        '--model-name', 'tiny_cnn',
        '--dir-type', 'deep',
        '--target-dir-name', 'ready',
    ]

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_dummy_weight(self):
        args_ = self.args + ['--fig-save-path', str(self.workdir / 'fig.png')]
        subprocess.run(args_, check=True)

    @pytest.mark.interactive
    def test_dummy_weight_with_plot(self):
        args_ = self.args
        subprocess.run(args_, check=True)
