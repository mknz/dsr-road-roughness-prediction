'''Evaluate module'''
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from road_roughness_prediction.tools import calc_plot_confusion_matrix


def evaluate(net, loader: DataLoader, class_names,
             epoch=None, writer=None, group=None, fig_save_path=None,
             device='cpu'):
    '''Evaluate trained model, optionally write result using TensorboardX'''

    n_class = len(class_names)

    class_count = [0 for _ in range(n_class)]
    class_correct = [0 for _ in range(n_class)]
    y_test = []
    y_pred = []

    net.eval()
    loss = 0.

    with torch.no_grad():
        for X, labels in loader:
            X = X.to(device)
            labels = labels.to(device)

            outputs = net.forward(X)
            loss += F.cross_entropy(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            labels_ = labels.tolist()
            predicted_ = predicted.tolist()

            y_test += labels_
            y_pred += predicted_

            for label, pred in zip(labels_, predicted_):
                class_count[int(label)] += 1
                class_correct[int(label)] += int(pred == label)

    loss /= len(loader.dataset)
    calc_plot_confusion_matrix(
        y_test, y_pred,
        class_names, writer=writer, group=group, epoch=epoch,
        fig_save_path=fig_save_path
    )

    accuracy = sum(class_correct) / sum(class_count)
    class_accuracy = [
        correct / count if count > 0 else 0.
        for correct, count
        in zip(class_correct, class_count)
    ]

    print(f'---{group}---')
    print(f'loss: {loss:.4f}')
    print(f'accuracy: {accuracy}')
    print(f'class_accuracy: {class_accuracy}')
    print(f'class_count: {class_count}')
    print(f'class_correct: {class_correct}')

    # TensorboardX
    if writer:
        class_accuracy_dict = {name: acc for name, acc in zip(class_names, class_accuracy)}
        class_correct_dict = {name: cr for name, cr in zip(class_names, class_correct)}

        writer.add_scalar(f'{group}/loss', loss, epoch)
        writer.add_scalar(f'{group}/accuracy', accuracy, epoch)
        writer.add_scalars(f'{group}/class_accuracy', class_accuracy_dict, epoch)
        writer.add_scalars(f'{group}/class_correct', class_correct_dict, epoch)
