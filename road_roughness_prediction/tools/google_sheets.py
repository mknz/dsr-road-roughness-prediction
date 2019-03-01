'''Google Sheets Tools'''
import os
from pathlib import Path
import subprocess

import pandas as pd


def save_csv(url: str, save_path: Path, sheet_name: str, show_summary=False):
    '''Download a data sheet from Google Sheets and save to csv file'''
    sheet_url = f'{url}&sheet={sheet_name}'
    subprocess.run(('wget', '-o', '/dev/null', '-O', str(save_path), sheet_url), check=True)
    recordings = pd.read_csv(str(save_path))
    if show_summary:
        print(recordings.head())


def main():
    env_var = 'GOOGLE_SHEETS_URL'
    url = os.environ.get(env_var)
    assert url, f'Invalid {env_var}'

    csv_path = Path('/tmp/road_roughness.csv')
    save_csv(url, csv_path, 'recordings', show_summary=True)


if __name__ == '__main__':
    main()
