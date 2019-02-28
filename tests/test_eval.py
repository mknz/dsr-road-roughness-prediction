'''Test evaluation code'''
from pathlib import Path
import subprocess
import shutil
import tempfile

import pytest
import torch

from road_roughness_prediction.config import EvalConfig
from road_roughness_prediction import models


class TestEvaluation:

    categories = ['asphalt', 'grass']
    n_class = len(categories)
    config = EvalConfig()

    workdir = Path(tempfile.mkdtemp())

    # Create dummy weight
    net = models.TinyCNN(n_class)
    weight_path = workdir / 'dummy.pth'
    torch.save(net.state_dict(), weight_path)

    image_dir = Path('tests/resources/surfaces')

    args = [
        'python3',
        'scripts/eval.py',
        '--weight-path', str(weight_path),
        '--image-dir', str(image_dir),
        '--model-name', 'tiny_cnn',
        '--dir-type', 'deep',
        '--target-dir-name', 'ready',
        '--categories',
    ]
    args += categories

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_dummy_weight(self):
        args_ = self.args + ['--save-fig-path', str(self.workdir / 'fig.png')]
        subprocess.run(args_, check=True)

    @pytest.mark.interactive
    def test_dummy_weight_with_plot(self):
        args_ = self.args
        subprocess.run(args_, check=True)
