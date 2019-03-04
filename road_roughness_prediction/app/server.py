'''App server'''
from functools import wraps
import time

from flask import Flask
from flask import flash
from flask import request
from flask import redirect

from PIL import Image

import numpy as np

from road_roughness_prediction.inference import ModelInference
from road_roughness_prediction.datasets.surface_types import SurfaceBasicCategory


app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
model_inf = ModelInference()


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
        print(f'Elapsed: {elapsed:.4f} Average: {np.mean(times):.4f} Std: {np.std(times):.4f}')
        return retvals
    return _wrapper


@measure_time
def predict(img):
    prob = model_inf.predict(img)
    print(prob)
    results = []
    for cat in SurfaceBasicCategory:
        name = cat.name
        index = cat.value
        results.append(f'{name:20s} {prob[0, index]:.4f}')
    return '\n'.join(results)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file_ = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file_.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file_ and allowed_file(file_.filename):
            img = np.array(Image.open(file_))[:, :, :3]
            result_str = predict(img)
            return '''
            <!doctype html>
            <title>Result</title>
            <h1>Result</h1>
            <pre>{}</pre>
            </html>
            '''.format(result_str)

    return '''
    <!doctype html>
    <title>Upload image</title>
    <h1>Upload image</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Predict>
    </form>
    '''


if __name__ == '__main__':
    app.run(debug=True)
