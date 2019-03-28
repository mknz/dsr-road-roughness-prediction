'''Segmentator app server'''
from functools import wraps
import io
import time
import os
from urllib.parse import quote
from urllib.parse import urlparse
from base64 import b64encode


from flask import Flask
from flask import flash
from flask import request
from flask import redirect
from flask import make_response
from flask import render_template

import requests

from PIL import Image

from albumentations.augmentations.functional import center_crop
from albumentations.augmentations.functional import resize

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor

import numpy as np

from road_roughness_prediction.segmentation import logging
from road_roughness_prediction.segmentation.inference import SidewalkSegmentator
from road_roughness_prediction.segmentation.datasets import surface_types
from road_roughness_prediction.tools import torch as torch_tools
from road_roughness_prediction.segmentation import utils


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


class Config:
    def __init__(self):
        _key = os.environ.get('GOOGLE_MAP_API_KEY')
        assert _key
        self.GOOGLE_MAP_API_KEY = _key


config = Config()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def measure_time(func):
    times = []

    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.time()
        retvals = func(*args, **kwargs)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f'Elapsed: {elapsed:.4f} sec Average: {np.mean(times):.4f} sec Std: {np.std(times):.4f} sec')
        return retvals
    return _wrapper


@measure_time
def load_segmentator():
    sidewalk_detector_weight_path = './resources/segmentation/weight_binary.pth'
    surface_segmentator_weight_path = './resources/segmentation/weight_multi.pth'
    device = torch.device('cpu')
    segmentator = SidewalkSegmentator(
        sidewalk_detector_weight_path=sidewalk_detector_weight_path,
        surface_segmentator_weight_path=surface_segmentator_weight_path,
        device=device,
    )
    return segmentator


def preprocessing(image: np.array) -> torch.Tensor:
    height, width = 640, 640
    image = resize(image, height, width)
    image = torch_tools.imagenet_normalize(image)
    return to_tensor(image).squeeze().unsqueeze(0)


@measure_time
def predict(image, segmentator):
    images = preprocessing(image)
    segmented, sidewalk_mask, background_mask = segmentator.run(images)
    labels, counts = np.unique(segmented, return_counts=True)
    n_total = np.prod(segmented.shape)

    def _get_count(index):
        for label, count in zip(labels, counts):
            if label == index:
                return count
        return 0

    results = []
    for category in surface_types.SimpleCategory:
        name = category.name
        index = category.value
        count = _get_count(index)
        percent = 100 * count / n_total
        results.append(f'{name:20s} {count:09d} {percent:6.2f}%')

    summary = '\n'.join(results)

    out_image = postprocessing(images, segmented, sidewalk_mask, background_mask)
    return summary, out_image


def _blend_image(original, segmented):
    blend = Image.blend(original.convert('RGB'), segmented.convert('RGB'), alpha=0.2)
    return blend


def postprocessing(inputs, segmented, sidewalk_mask, background_mask):
    input_img = to_pil_image(logging.normalize(inputs[0, ::]))

    out_seg_img = segmented[0, ::]
    out_seg_index_img = utils.create_index_image(out_seg_img)

    blend_output_img = _blend_image(input_img, out_seg_index_img)

    img_ = (sidewalk_mask[0, ::] * 255).astype(np.uint8)
    img_ = Image.fromarray(img_)
    img_ = _blend_image(input_img, img_)

    return blend_output_img


def is_valid_url(url):

    MAX_LENGTH = 1000

    if not url:
        return False

    if len(url) > MAX_LENGTH:
        return False

    r = urlparse(url)

    if r.scheme == '':
        return False

    if r.netloc == '':
        return False

    return True


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get('SIDEWALK_APP_SECRET_KEY')

    segmentator = load_segmentator()

    @app.route('/', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            image_url = request.form.get('image_url')
            if not is_valid_url(image_url):
                flash('Invalid url', 'warning')
                return redirect(request.url)

            resp = requests.get(image_url)
            if resp.status_code != 200:
                flash('Cannot get image', 'warning')
                return redirect(request.url)

            with io.BytesIO() as buf:
                buf.write(resp.content)
                buf.seek(0)
                img = np.array(Image.open(buf))[:, :, :3]

            summary, out_image = predict(img, segmentator)
            bytes_ = utils.pil_image_to_bytes(out_image, format='JPEG')
            encoded_image = b64encode(bytes_).decode('ascii')
            out_image_url = f'data:image/jpeg;base64,{quote(encoded_image)}'

            return render_template(
                'prediction.html',
                out_image_url=out_image_url, summary=summary,
                config=config,
            )

            # check if the post request has the file part
            if 'image_file' not in request.files:
                flash('No file part', 'warning')
                return redirect(request.url)
            file_ = request.files['image_file']

            # if user does not select file, browser also
            # submit an empty part without filename
            if file_.filename == '':
                flash('No selected file', 'warning')
                return redirect(request.url)

            if file_ and allowed_file(file_.filename):
                img = np.array(Image.open(file_))[:, :, :3]
                summary, out_image = predict(img, segmentator)
                bytes_ = utils.pil_image_to_bytes(out_image, format='JPEG')
                encoded_image = b64encode(bytes_).decode('ascii')
                out_image_url = f'data:image/jpeg;base64,{quote(encoded_image)}'
                return render_template(
                    'prediction.html',
                    out_image_url=out_image_url, summary=summary,
                    config=config,
                )
        return render_template('index.html', config=config)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
