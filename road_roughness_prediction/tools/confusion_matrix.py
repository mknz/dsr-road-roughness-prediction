'''Confusion Matrix
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html'''
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..tools.image_utils import fig_to_pil

plt.rcParams.update({'font.size': 18})

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig_size = (12, 10)
    fig = plt.figure(figsize=fig_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return fig


def _prep_label(y_test, y_pred, class_names):
    '''Remove zero instance labels and reindex y'''
    use_labels = sorted(list(set(y_test + y_pred)))
    use_class_names = [class_names[i] for i in use_labels]

    order = {}
    class_names_ = []
    for i, (prev, name) in enumerate(zip(use_labels, use_class_names)):
        order[prev] = i
        class_names_.append(name)
    y_test_ = [order[x] for x in y_test]
    y_pred_ = [order[x] for x in y_pred]
    return y_test_, y_pred_, class_names_


def calc_plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        fig_save_path: Path = None,
        writer=None,
        group=None,
        epoch=None,
        show_plot=False
):
    '''Calculate and plot confusion matrix'''

    y_test_, y_pred_, class_names_ = _prep_label(y_test, y_pred, class_names)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_, y_pred_)
    np.set_printoptions(precision=2)

    def _save(fig, path, cond: str):
        filename = path.stem + cond + path.suffix
        save_path = path.parent / filename
        fig.savefig(save_path)


    def _handle_output(fig, fig_save_path, writer, cond, group, epoch, show_plot):
        if fig_save_path:
            _save(fig, fig_save_path, f'_{cond}')

        if writer:
            img = np.array(fig_to_pil(fig)).astype(np.uint8)
            img_ = img[:, :, :3]  # Remove alpha
            writer.add_image(f'{group}/cm/{cond}', img_, epoch, dataformats='HWC')

        if show_plot:
            fig.show()

    # Plot non-normalized confusion matrix
    fig = plot_confusion_matrix(
        cnf_matrix,
        classes=class_names_,
        title='Confusion matrix, without normalization',
    )

    _handle_output(fig, fig_save_path, writer, 'non_norm', group, epoch, show_plot)

    # Plot normalized confusion matrix
    fig = plot_confusion_matrix(
        cnf_matrix,
        classes=class_names_,
        normalize=True,
        title='Normalized confusion matrix',
    )

    _handle_output(fig, fig_save_path, writer, 'with_norm', group, epoch, show_plot)
