'''Confusion Matrix
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html'''
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


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

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


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


def calc_plot_confusion_matrix(y_test, y_pred, class_names, save_path: Path):

    _FIG_SIZE = (12, 10)

    y_test_, y_pred_, class_names_ = _prep_label(y_test, y_pred, class_names)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_, y_pred_)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=_FIG_SIZE)
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names_,
        title='Confusion matrix, without normalization',
    )

    def _save(path, cond:str):
        filename = path.stem + cond + path.suffix
        save_path = path.parent / filename
        plt.savefig(save_path)

    if save_path:
        _save(save_path, '_without_norm')

    # Plot normalized confusion matrix
    plt.figure(figsize=_FIG_SIZE)
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names_,
        normalize=True,
        title='Normalized confusion matrix',
    )

    if save_path:
        _save(save_path, '_with_norm')

    if not save_path:
        plt.show()
