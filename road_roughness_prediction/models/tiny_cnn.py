'''Basic CNN'''
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Dropout(0.2),
    )


class TinyCNN(nn.Module):
    '''Tiny convolutional network for testing purpose'''
    _fc1_input_size = 128 * 16 * 16

    def __init__(self, n_out):
        super(TinyCNN, self).__init__()
        self.conv1 = conv_layer(3, 16)
        self.conv2 = conv_layer(16, 32)
        self.conv3 = conv_layer(32, 64)
        self.conv4 = conv_layer(64, 128)

        self.fc1 = nn.Sequential(
            nn.Linear(self._fc1_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Linear(512, n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self._fc1_input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
