'''Evaluation module'''

import torch
from torch.utils.data import DataLoader


def evaluate(
        net,
        loader: DataLoader,
        epoch,
        criterion,
        device,
        logger=None,
        group=None,
):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    net.eval()
    loss = 0.

    with torch.no_grad():
        for i, batch in enumerate(loader):
            X = batch['X']
            Y = batch['Y']
            X = X.to(device)
            Y = Y.to(device)
            out = net.forward(X)

            if criterion:
                loss += criterion(out, Y).item()

            if i == 0:
                first_out = out
                first_batch = batch

    if criterion:
        loss /= len(loader.dataset)
        print(f'{group} loss: {loss:.4f}')

    # First epoch
    if epoch == 1 and logger:
        logger.add_images_from_path(f'{group}/images', first_batch['image_path'])
        logger.add_masks_from_path(f'{group}/masks', first_batch['mask_path'])
        logger.add_input(f'{group}/inputs', first_batch['X'].cpu())
        logger.add_target(f'{group}/targets', first_batch['Y'].cpu())

    # Every epoch
    if logger:
        logger.writer.add_scalar(f'{group}/loss', loss, epoch)
        logger.add_output(f'{group}/outputs', first_out.cpu(), epoch)
