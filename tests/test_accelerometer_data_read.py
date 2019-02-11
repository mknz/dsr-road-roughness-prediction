from pathlib import Path

import matplotlib.pyplot as plt

import road_roughness_prediction.tools.accelerometer as acc

TEST_DATA_PATH = './tests/one_second_flip.txt'

def test_read_data():
    reader = acc.DataReader(Path(TEST_DATA_PATH), acc.preprocess)

    plot_range = range(1000)

    def _plot(df, title):
        fig = plt.figure()
        fig.suptitle(title)
        for col in ['X', 'Y', 'Z']:
            plt.plot(df.iloc[plot_range]['time'], df.iloc[plot_range][col], label=col)
        plt.legend()
        plt.show()

    raw_df = reader.get_raw_df()
    _plot(raw_df, 'Raw')

    df = reader.get_df()
    _plot(df, 'Prep')

    assert len(df) == len(raw_df)
