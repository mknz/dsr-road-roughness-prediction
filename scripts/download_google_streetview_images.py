#!/usr/bin/env/python3
import argparse
import os
from pathlib import Path

from road_roughness_prediction.tools.google_street_view import StreetView


def main():
    '''Read filepath, latitude, longitude from csv file and get Google StreetView images'''
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('--savedir', default='google_street_view')
    args = parser.parse_args()
    csv_file_path = args.filepath
    save_dir = Path(args.savedir)

    HEADINGS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    GOOGLE_MAP_API_KEY = os.environ.get('GOOGLE_MAP_API_KEY')
    assert GOOGLE_MAP_API_KEY, '$GOOGLE_MAP_API_KEY was not found in env variable'

    with open(csv_file_path, 'r') as csv_file:
        lines = csv_file.readlines()

    street_view = StreetView(GOOGLE_MAP_API_KEY)
    for line in lines:
        file_path, latitude, longitude = line.split(',')
        file_path = Path(file_path)
        latitude = float(latitude)
        longitude = float(longitude)

        # Assume directory structure $CATEGORY/$ID/$VIDEOFILE
        category = file_path.parent.parent

        # Save to category-wise directories
        save_category_dir = save_dir / category
        save_category_dir.mkdir(exist_ok=True, parents=True)
        basename = file_path.stem

        # Get images for each heading angle
        for heading in HEADINGS:
            save_path = save_category_dir / f'{basename}_{heading:03d}.jpg'
            street_view.save_image(save_path, latitude, longitude, heading=heading)


if __name__ == '__main__':
    main()
