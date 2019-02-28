'''Test evaluation code'''
import os
from pathlib import Path
import subprocess

import torch

from road_roughness_prediction.config import EvalConfig
from road_roughness_prediction import models


class TestEvaluation:
    categories = ['asphalt', 'grass']
    n_class = len(categories)
    config = EvalConfig()

    # Create dummy weight
    net = models.TinyCNN(n_class)
    weight_path = Path('/tmp/dummy.pth')
    torch.save(net.state_dict(), weight_path)

    image_dir = Path('tests/resources/surfaces')

    def teardown(self):
        # Delete dummy weight
        os.remove(Path(self.weight_path))

    def test_dummy_weight(self):
        args = [
            'python3',
            'scripts/eval.py',
            '--weight-path', str(self.weight_path),
            '--image-dir', str(self.image_dir),
            '--model-name', 'tiny_cnn',
            '--dir-type', 'deep',
            '--target-dir-name', 'ready',
            '--categories'
        ]
        args += self.categories
        subprocess.run(args, check=True)
