'''Image utils'''
from io import BytesIO

from PIL import Image
import matplotlib.pyplot as plt


def matplot_to_pil(fig: plt.Figure):
    '''Convert matplot figure to PIL Image'''
    bio = BytesIO()
    fig.savefig(bio)
    bio.seek(0)
    byte_img = bio.read()
    bio_img = BytesIO(byte_img)
    return Image.open(bio_img)
