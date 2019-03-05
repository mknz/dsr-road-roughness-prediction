'''Model inference'''
import torch

from road_roughness_prediction.config import Config
from road_roughness_prediction import models
from road_roughness_prediction.datasets.transformations import TransformFactory
from road_roughness_prediction.datasets.transformations import TransformType


class ModelInference:
    def __init__(self):
        self.config = Config()
        self.config.from_dict(dict(TRANSFORMATION=TransformType.BASIC_EVAL_TRANSFORM))
        self._transform = TransformFactory(self.config)

        model_name = 'resnet18'
        weight_path = 'data/weights/surface_category_classification/resnet18/extensive_transform/resnet18_dict_epoch_016.pth'
        n_class = 8

        if model_name == 'tiny_cnn':
            self.net = models.TinyCNN(n_class)
        elif model_name == 'resnet18':
            self.net = models.Resnet18(n_class)

        self.net.load_state_dict(state_dict=torch.load(weight_path))
        print(f'{model_name} loaded.')

    def _prep(self, image):
        return self._transform.full_transform(image=image)['image'].unsqueeze(0)

    def predict(self, image):
        input_tensor = self._prep(image)
        self.net.eval()
        with torch.no_grad():
            out = self.net.forward(input_tensor)
            prob = torch.nn.functional.softmax(out, dim=1)
        return prob.detach().numpy()
