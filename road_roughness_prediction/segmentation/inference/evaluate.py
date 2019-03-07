'''Evaluate module'''
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate(net, loader: DataLoader, epoch=None, writer=None, group=None):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    net.eval()
    loss = 0.
    outputs = []
    with torch.no_grad():
        for X, Y in loader:
            out = net.forward(X)
            loss += F.binary_cross_entropy_with_logits(out, Y)
            outputs.append(out)

    outputs = torch.cat(outputs, dim=0)
    loss /= len(loader.dataset)

    print(f'loss: {loss:.4f}')

    # TensorboardX
    if writer:
        writer.add_scalar(f'{group}/loss', loss, epoch)
        write_images = torch.cat([outputs, outputs, outputs], dim=1)
        writer.add_images(f'{group}/out', write_images, epoch)
