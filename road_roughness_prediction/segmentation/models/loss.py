'''https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py'''
import torch
from torch import nn

from road_roughness_prediction.segmentation.datasets import surface_types


def get_criterion(category_type, jaccard_weight=None, class_weights=None):
    if category_type == surface_types.BinaryCategory:
        criterion = LossBinary(jaccard_weight)
    else:
        criterion = LossMulti(jaccard_weight, class_weights, num_classes=len(category_type))
    return criterion


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        self.nll_loss = nn.NLLLoss(weight=class_weights)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        eps = 1e-15
        for cls in range(self.num_classes):
            jaccard_target = (targets == cls).float()
            jaccard_output = outputs[:, cls].exp()
            intersection = (jaccard_output * jaccard_target).sum()

            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss
