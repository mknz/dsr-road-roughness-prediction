'''Test training'''
from pathlib import Path
import subprocess
import shutil
import tempfile


class TestTraining:

    workdir = Path(tempfile.mkdtemp())

    image_dir = Path('tests/resources/segmentation/labelme/JPEGImages')
    mask_dir = Path('tests/resources/segmentation/labelme/SegmentationClassPNG')
    args = [
        'python3',
        'train_seg.py',
        '--train-image-dirs', str(image_dir), str(image_dir), str(image_dir),
        '--train-mask-dirs', str(mask_dir), str(mask_dir), str(mask_dir),
        '--train-dataset-types', 'base', 'base', 'base',
        '--validation-image-dirs', str(image_dir), str(image_dir), str(image_dir),
        '--validation-mask-dirs', str(mask_dir), str(mask_dir), str(mask_dir),
        '--validation-dataset-types', 'base', 'base', 'base',
        '--input-size', '64', '64',
        '--batch-size', '128',
        '--epochs', '2',
        '--save-dir', str(workdir),
        '--run-name', 'test',
    ]

    def teardown_class(self):
        # Delete temp dir and all of its content
        shutil.rmtree(self.workdir)

    def test_unet11_binary(self):
        args_ = self.args + ['--model-name', 'unet11']
        subprocess.run(args_, check=True, timeout=60)

    def test_unet11_binary(self):
        args_ = self.args + ['--model-name', 'unet11']
        args_ += ['--cpu']
        subprocess.run(args_, check=True, timeout=60)

    def test_unet11_simple(self):
        args_ = self.args + ['--model-name', 'unet11']
        args_ += ['--category-type', 'simple']
        subprocess.run(args_, check=True, timeout=60)

    def test_unet16_simple(self):
        args_ = self.args + ['--model-name', 'unet16']
        args_ += ['--category-type', 'simple']
        subprocess.run(args_, check=True, timeout=60)
