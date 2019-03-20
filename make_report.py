'''Read output images, produce report'''
import argparse
from pathlib import Path

import numpy as np

from PIL import Image

from road_roughness_prediction.segmentation.datasets import surface_types


def compute_confusion_matrix(true, pred, n_classes):
    result = np.zeros((n_classes, n_classes))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def build_confusion_matrix(ground_trouth_batch, prediction_batch, category_type):
    n_classes = len(category_type)
    conf_matrix = np.zeros((n_classes, n_classes))

    # loop over each tensor in batch
    for b in range(ground_trouth_batch.shape[0]):
        conf_matrix += compute_confusion_matrix(
            ground_trouth_batch[b, ::].flatten(),
            prediction_batch[b, ::].flatten(),
            n_classes,
        )
    return conf_matrix


def calc_metrics(confusion_matrix, category_id = -1):
    if category_id == -1:  # overall metrics
        true_pos = np.sum(np.diag(confusion_matrix))
        false_pos = np.sum(confusion_matrix) - true_pos
        accuracy = precision = recall = f_measure = jaccard_index = true_pos / (true_pos + false_pos + 1e-15)
    else:
        true_pos  = confusion_matrix[category_id, category_id]
        true_neg  = np.sum(np.diag(confusion_matrix)) - true_pos
        false_pos = np.sum(confusion_matrix, axis=0)[category_id]  # column for category_id
        false_neg = np.sum(confusion_matrix, axis=1)[category_id]  # row for category_id

        accuracy  = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-15)
        precision = (true_pos) / (true_pos + false_pos + 1e-15)
        recall    = (true_pos) / (true_pos + false_neg + 1e-15)
        f_measure = (2 * recall * precision) / (recall + precision + 1e-15)
        jaccard_index = true_pos / (true_pos + false_pos + false_neg + 1e-15)

    return accuracy, precision, recall, f_measure, jaccard_index


def load(data_dir: Path):
    target_dir = data_dir / 'target'
    output_dir = data_dir / 'output'
    assert target_dir.exists()
    assert output_dir.exists()

    targets, outputs = [], []
    for target_path in target_dir.glob('*.png'):
        output_path = output_dir / target_path.name
        assert output_path.exists()
        target = np.array(Image.open(target_path))
        output = np.array(Image.open(output_path))
        targets.append(target)
        outputs.append(output)
    return np.array(targets), np.array(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', default='./report')
    parser.add_argument('--category-type', default='simple')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    category_type = surface_types.from_string(args.category_type)

    target, output = load(data_dir)
    confusion_matrix = build_confusion_matrix(target, output, category_type)
    np.set_printoptions(formatter={'all': lambda x: f'{x:08.0f}'})

    for category in category_type:
        print(f'{category.name:15s}{category.value:02d}')
        metrics = calc_metrics(confusion_matrix, category.value)
        print(metrics)

    print('confusion_matrix')
    print(confusion_matrix)

    metrics = calc_metrics(confusion_matrix)
    print('all', metrics)

    metrics = calc_metrics(confusion_matrix[1:, 1:])
    print('sidwalk', metrics)



if __name__ == '__main__':
    main()
