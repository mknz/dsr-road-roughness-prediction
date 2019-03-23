'''Models'''
from road_roughness_prediction.segmentation.datasets import surface_types

from .unet import UNet11
from .unet import UNet16
from .unet import AlbuNet
from .loss import LossBinary


def load_model(model_name: str, category_type: surface_types.SurfaceCategoryBase):
    if model_name == 'unet11':
        if category_type == surface_types.BinaryCategory:
            net = UNet11(pretrained=True)
        else:
            net = UNet11(num_classes=len(category_type), pretrained=True)
    elif model_name == 'unet16':
        if category_type == surface_types.BinaryCategory:
            net = UNet16(pretrained=True)
        else:
            net = UNet16(num_classes=len(category_type), pretrained=True)
    else:
        raise ValueError(model_name)

    return net
