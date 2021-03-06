'''Segmentator app server'''
from functools import wraps
import io
import os
import re
import time
from urllib.parse import quote
from urllib.parse import urlparse
from base64 import b64encode


from flask import Flask
from flask import flash
from flask import request
from flask import redirect
from flask import render_template

import requests

from PIL import Image

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

    out_image = postprocessing(images, segmented, sidewalk_mask, background_mask)

    summary = []
    for category in surface_types.SimpleCategory:
        result = {}

        name = category.name
        result['name'] = name

        index = category.value
        count = _get_count(index)
        result['count'] = count

        result['percent'] = 100 * count / n_total
        summary.append(result)

    summary.sort(key=lambda x: x['percent'], reverse=True)
    return summary, out_image


def _blend_image(original, segmented):
    blend = Image.blend(original.convert('RGB'), segmented.convert('RGB'), alpha=0.2)
    return blend


def postprocessing(inputs, segmented, sidewalk_mask, background_mask):
    input_img = to_pil_image(logging.normalize(inputs[0, ::]))

    out_seg_img = segmented[0, ::]
    out_seg_index_img = utils.create_index_image(out_seg_img)

    blend_output_img = _blend_image(input_img, out_seg_index_img).convert('RGBA')

    legend = Image.open('./road_roughness_prediction/app/static/legend_subset.png')
    blend_output_img.paste(legend, (10, 10))

    return blend_output_img.convert('RGB')


def is_valid_url(url: str):
    '''Validate url string'''

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


def convert_gs_url(url, api_key):
    '''If given url is a Google Street View URL, convert it into a statc image URL'''
    match = re.search('https://www.google.*/maps/@([\d|\.]*),([\d|\.]*),3a,([\d|\.]*)y,([\d|\.]*)h,([\d|\.]*)t', url)
    if not match:
        return url

    lat = float(match.group(1))
    lng = float(match.group(2))
    fov = float(match.group(3))
    heading = float(match.group(4))
    pitch = float(match.group(5)) - 90

    converted_url = f'https://maps.googleapis.com/maps/api/streetview?size=640x640&fov={fov}&pitch={pitch}&location={lat},{lng}&heading={heading}&key={api_key}'
    return converted_url


class UploadFileGetError(Exception):
    pass


class ImageFileGetError(Exception):
    pass


def _get_uploaded_image_file(request):
    file_ = request.files.get('image_file')
    if not file_:
        raise UploadFileGetError('No file part')
    if not allowed_file(file_.filename):
        raise UploadFileGetError('Not an allowed file')
    try:
        img = np.array(Image.open(file_))[:, :, :3]
    except:
        raise UploadFileGetError('Could not open uploaded file')

    return img

def _get_image_file_from_url(request, config):
    image_url = request.form.get('image_url')
    if not image_url:
        raise ImageFileGetError('No image url in request')

    # Basic ur validation
    if not is_valid_url(image_url):
        raise ImageFileGetError('Invailid URL')

    # Try to convert to static image
    try:
        image_url = convert_gs_url(image_url, config.GOOGLE_MAP_API_KEY)
    except:
        raise ImageFileGetError('Something wrong with the URL')

    # Get image from URL
    resp = requests.get(image_url)
    if resp.status_code != 200:
        raise ImageFileGetError('Invalid respose code')

    # Read image as np array
    try:
        with io.BytesIO() as buf:
            buf.write(resp.content)
            buf.seek(0)
            img = np.array(Image.open(buf))[:, :, :3]
    except:
        raise ImageFileGetError('File open error')

    return img


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get('SIDEWALK_APP_SECRET_KEY')

    segmentator = load_segmentator()

    @app.route('/', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            try:
                img = _get_uploaded_image_file(request)
            except UploadFileGetError:
                try:
                    img = _get_image_file_from_url(request, config)
                except ImageFileGetError as error:
                    #flash(str(error), 'warning')
                    flash('Could not get image', 'warning')
                    return render_template('index.html', config=config)

            # Make a prediction
            summary, out_image = predict(img, segmentator)

            # Encode segmented image
            bytes_ = utils.pil_image_to_bytes(out_image, format='JPEG')
            encoded_image = b64encode(bytes_).decode('ascii')
            out_image_url = f'data:image/jpeg;base64,{quote(encoded_image)}'

            return render_template(
                'prediction.html',
                out_image_url=out_image_url,
                summary=summary,
                config=config,
            )

        return render_template('index.html', config=config)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
