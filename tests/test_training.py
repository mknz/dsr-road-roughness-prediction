'''Test training'''
from pathlib import Path
import subprocess
import shutil
import tempfile

from road_roughness_prediction.config import Config
from road_roughness_prediction.datasets.transformations import TransformType


class TestTraining:

    categories = ['asphalt', 'grass']
    n_class = len(categories)

    config = Config()
    config.from_dict(dict(TRANSFORMATION=TransformType.BASIC_EVAL_TRANSFORM))

    workdir = Path(tempfile.mkdtemp())

    image_dir = Path('tests/resources/surfaces')

    args = [
        'python3',
        'scripts/train.py',
        '--data-dir', str(image_dir),
        '--target-dir-name', 'ready',
        '--batch-size', '128',
        '--epochs', '1',
        '--class-balanced',
        '--transform', 'extensive_transform',
        '--save-dir', str(workdir),
    ]
    args.append('--categories')
    args += categories

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_tiny_cnn(self):
        args_ = self.args + ['--model-name', 'tiny_cnn']
        subprocess.run(args_, check=True)

    def test_resnet18(self):
        args_ = self.args + ['--model-name', 'resnet18']
        subprocess.run(args_, check=True)
