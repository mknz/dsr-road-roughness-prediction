import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir')
    parser.add_argument('--mask-dir')
    args = parser.parse_args()
    mask_dir = Path(args.mask_dir)
    print('mask_path,sidewalk_pixels')
    for mask_path in mask_dir.glob('*.png'):
        mask = cv2.imread(str(mask_path))
        sidewalk_pixels = mask[mask == 1].sum()
        print(f'{str(mask_path)},{sidewalk_pixels}')


if __name__ == '__main__':
    main()
