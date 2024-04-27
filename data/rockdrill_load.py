'''
Fault Classification in Rock Drills from 2022 PHM Conference Data Challenge.
https://www.kaggle.com/datasets/erikjakobsson/fault-classification-in-rock-drills
'''
import logging
import re

import numpy as np
import pandas as pd


def _load_raw_data(path, data_type='training'):
    '''
    Load raw data from the path and return a dictionary with the following
    structure:
    {
        'file_name': {
            'X': np.array,
            'y': np.array,
            'Sensor': str,
            'Setting': str
        },
        ...
    }

    Parameters
    ----------
    path : Path
        Path to the directory with the raw data.
    data_type : str
        Which data to load (training, validation, test, ref).

    Returns
    -------
    data : dict
    '''

    data_dir = path / data_type

    files = sorted(list(data_dir.glob('*.csv')))
    data = {}

    for file in files:
        data[file.stem] = {}
        # Needed because all rows have different length
        df_tmp = pd.read_csv(
            file, header=None, sep='/n', engine='python'
        )[0].str.split(',', expand=True)
        pattern_match = re.search(r'_([a-zA-Z]+)(\d+)$', file.stem)
        data[file.stem]['X'] = df_tmp.iloc[:, 1:].values.astype(float)
        data[file.stem]['Sensor'] = pattern_match.group(1)
        data[file.stem]['Setting'] = int(pattern_match.group(2))

        if 'data' in file.stem:
            file_ans = file.parent.parent / 'answers' / \
                f"ans{data[file.stem]['Setting']}.csv"
            data[file.stem]['y'] = pd.read_csv(
                file_ans, header=None
            ).values.squeeze() - 1
        elif 'ref' in file.stem:
            data[file.stem]['y'] = df_tmp.iloc[:, 0].values.astype(int) - 1
        else:
            raise ValueError(f'Unknown file: {file}')

    return data


def _stack_signals(data_dict, sensors=['pdmp', 'pin', 'po'], max_cols=None):
    '''
    Stack signals from different sensors to one array per sensor.
    The signals are padded to the same length.

    Parameters:
    -----------
    data_dict: dict
        Dictionary with raw data
    sensors: list
        List of sensors to stack signals for
    max_cols: int
        Maximum number of columns in the signals. If None, the maximum
        number of columns in the signals is used.

    Returns
    -------
    X: dict
        Dictionary with stacked signals
    y: np.array
        Array with labels
    settings: np.array
        Array with settings
    '''
    data_sensor_ind = [
        k for k in data_dict.keys() if any(sensor in k for sensor in sensors)
    ]
    X = {k: [] for k in sensors}
    y = []
    settings = []
    for dsi in data_sensor_ind:
        y.append(data_dict[dsi]['y'])
        settings.extend([data_dict[dsi]['Setting']] * len(data_dict[dsi]['y']))
        for s in sensors:
            tmp_key = f"{dsi.split('_')[0]}_{s}{settings[-1]}"
            X[s].append(data_dict[tmp_key]['X'])
    y = np.concatenate(y)
    settings = np.array(settings)

    # Pad or truncate signals to same length and stack to one array per sensor
    for k, v in X.items():
        if max_cols is None:
            max_cols = max([sig.shape[1] for sig in v])
        res = []
        for sig in v:
            if max_cols > sig.shape[1]:
                # Pad signals
                padded_sig = np.pad(
                    sig,
                    ((0, 0), (0, max_cols - sig.shape[1])),
                    mode='constant',
                    constant_values=np.nan
                )
            else:
                # Truncate signals
                padded_sig = sig[:, :max_cols]
            res.append(padded_sig)
        X[k] = np.vstack(res)

    return X, y, settings


def load(path, data_type='training', max_cols=None, force_reload=False):
    '''
    Load data from Rock Drill Diagnostic Challenge.

    Parameters
    ----------
    path : Path
        Path to the directory with the raw data.
    data_type : str
        Which data to load (training, validation, test, ref).
    max_cols : int
        Maximum number of columns in the signals. If None, the maximum
        number of columns in the signals is used.
    force_reload : bool
        Whether to reload data from source or use saved data.

    Returns
    -------
    X : dict
        Dictionary with stacked signals
    y : array
        Array with labels
    settings : array
        Array with settings
    '''
    saved_dir = path / data_type / f'saved_{max_cols}'
    if saved_dir.exists() and not force_reload:
        logging.info('Loading data from %s', saved_dir)
        X = {
            s: np.load(saved_dir / f'X_{s}.npy')
            for s in ['pdmp', 'pin', 'po']
        }
        y = np.load(saved_dir / 'y.npy')
        settings = np.load(saved_dir / 'settings.npy')
        return X, y, settings

    logging.info('Loading data from source %s', path / data_type)
    data = _load_raw_data(path, data_type)
    X, y, settings = _stack_signals(data, max_cols=max_cols)

    saved_dir.mkdir(parents=True, exist_ok=True)
    for s in X.keys():
        np.save(saved_dir / f'X_{s}.npy', X[s])
    np.save(saved_dir / 'y.npy', y)
    np.save(saved_dir / 'settings.npy', settings)

    return X, y, settings
