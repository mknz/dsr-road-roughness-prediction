import torch.nn as nn
import torchvision


class Resnet18(nn.Module):

    def __init__(self, n_out):
        super(Resnet18, self).__init__()
        self.base_net = torchvision.models.resnet18(pretrained=True)
        for param in self.base_net.parameters():
            param.requires_grad = False

        self.base_net.fc = nn.Linear(2048, n_out)

    def forward(self, x):
        return self.base_net(x)
