'''Get images from Google Street View'''
from pathlib import Path
import shutil

import requests


class InvalidResponse(Exception):
    pass


class StreetView:
    ''' Get image from Google StreetView '''

    def __init__(self, api_key):
        self.api_key = api_key

    def save_image(
            self,
            image_path: Path,
            latitude: float,
            longitude: float,
            heading: int = 0,
            pitch: int = 0,
            fov: int = 120,
            width: int = 640,
            height: int = 640,
    ):
        # Construct api url
        image_size = f'{width}x{height}'
        url = f'https://maps.googleapis.com/maps/api/streetview?location={latitude},{longitude}&size={image_size}&heading={heading}&pitch={pitch}&fov={fov}&key={self.api_key}'

        # Get response and save
        resp = requests.get(url, stream=True)
        if resp.status_code == 200:
            with image_path.open('wb') as f:
                resp.raw.decode_content = True
                shutil.copyfileobj(resp.raw, f)
            print(f'Saved {str(image_path)}')
        else:
            raise InvalidResponse(f'{url}, statue_code: {resp.status_code}')
