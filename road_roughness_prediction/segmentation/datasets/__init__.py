from ._datasets import SidewalkSegmentationDataset
from ._datasets import SidewalkSegmentationDatasetFactory
from ._bdd import BddDataset
from ._bdd import BddDatasetFactory


def create_dataset(dataset_type:str, image_dirs, mask_dirs, category_type, transform):

    if dataset_type in ['base', 'walk', 'misc']:
        dataset = SidewalkSegmentationDatasetFactory(
            image_dirs,
            mask_dirs,
            category_type,
            transform,
        )
    elif dataset_type == 'bdd':
        dataset = BddDatasetFactory(
            image_dirs,
            mask_dirs,
            category_type,
            transform,
        )
    else:
        raise ValueError(dataset_type)

    return dataset
