'''Image utils'''
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt


def save_and_open(save_func):
    '''Save to in-memory buffer and re-open '''
    buf = BytesIO()
    save_func(buf)
    buf.seek(0)
    bytes_ = buf.read()
    buf_ = BytesIO(bytes_)
    return buf_


def matplot_to_pil(fig: plt.Figure):
    '''Convert matplot figure to PIL Image'''
    buf = save_and_open(fig.savefig)
    return Image.open(buf)
