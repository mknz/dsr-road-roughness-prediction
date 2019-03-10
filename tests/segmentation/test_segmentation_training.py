'''Test training'''
from pathlib import Path
import subprocess
import shutil
import tempfile


class TestTraining:

    workdir = Path(tempfile.mkdtemp())

    image_dir = Path('tests/resources/segmentation/labelme')
    args = [
        'python3',
        'train_seg.py',
        '--train-data-dirs', str(image_dir), str(image_dir),  str(image_dir),
        '--validation-data-dirs', str(image_dir), str(image_dir),  str(image_dir),
        '--input-size', '64', '64',
        '--batch-size', '128',
        '--epochs', '2',
        '--save-dir', str(workdir),
        '--run-name', 'test',
        '--cpu',
    ]

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_unet11(self):
        args_ = self.args + ['--model-name', 'unet11']
        subprocess.run(args_, check=True, timeout=120)
