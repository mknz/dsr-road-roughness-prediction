'''Accelerometer data processing tools
http://www.keuwl.com/Accelerometer/
'''
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class DataReader:
    '''Data reader class'''
    def __init__(self, path: Path, prep_func: Callable) -> None:
        self._raw_df = rename_column_names(self._read_data(path))
        self._df = prep_func(self._raw_df)

    @staticmethod
    def _read_data(path: Path):
        '''Read data from csv'''
        with path.open('r') as f:
            # Skip header lines
            for _ in range(3):
                f.readline()
            df = pd.read_csv(f)
        return df

    def get_raw_df(self):
        '''Get raw data frame'''
        return self ._raw_df.copy()

    def get_df(self):
        '''Get preprocessed data frame'''
        return self._df.copy()


def preprocess(df: pd.DataFrame, data_range=(-3.0, 3.0)):
    '''Preprocess and returns new data frame'''
    df_ = df.copy()
    minmax_scaler = MinMaxScaler(data_range)
    standard_scaler = StandardScaler()
    columns = ['X', 'Y', 'Z', 'R']
    df_[columns] = minmax_scaler.fit_transform(df_[columns])
    df_[columns] = standard_scaler.fit_transform(df_[columns])
    return df_


def rename_column_names(df: pd.DataFrame):
    '''Rename column names'''
    return df.rename(index=str, columns={
        'Time (s)': 'time',
        ' X (m/s2)': 'X',
        ' Y (m/s2)': 'Y',
        ' Z (m/s2)': 'Z',
        ' R (m/s2)': 'R',
        ' Theta (deg)': 'Theta',
        ' Phi (deg)': 'Phi',
    })
