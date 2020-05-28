import array
import os

import numpy as np

from Preprocessor import Preprocessor, ClassificationPreprocessor, AutoencoderPreprocessor

preprocessors = {
    'base': Preprocessor,
    'classification': ClassificationPreprocessor,
    'autoencoder': AutoencoderPreprocessor
}


def recognize_type(filename):
    parts = filename.split('.')
    if len(parts) == 2:
        return parts[-1]
    else:
        raise ValueError("unrecognized file type")


def read_uci_file(filename):
    with open(os.path.join("DataSets", filename)) as file:
        table = np.array(
            [[float(f) for f in row.split(',')] for row in file.read().splitlines() if row.find('?') == -1]
        )
    return table


def read_bin_file(filename):
    with open(os.path.join("DataSets", filename), "rb") as file:
        file.seek(0)
        n = int.from_bytes(file.read(4), byteorder='little')
        in_dim = int.from_bytes(file.read(4), byteorder='little')
        out_dim = int.from_bytes(file.read(4), byteorder='little')
        data = file.read()
    print('n: {}, in_dim: {}, out_dim: {}'.format(n, in_dim, out_dim))
    row_size = (in_dim + out_dim) * 8
    rows = [data[i:i+row_size] for i in range(0, len(data), row_size)]
    table = np.array([array.array('d', row) for row in rows])
    return table


files_readers = {
    'uci': read_uci_file,
    'bin': read_bin_file,
}


def read_file(file_name,
              mode,
              input_dim=None,
              output_dim=None,
              **kwargs):
    table = files_readers[recognize_type(file_name)](file_name)
    sample_size = len(table[0])
    if output_dim is None:
        output_dim = 1
    if input_dim is None:
        input_dim = sample_size - output_dim
    elif input_dim < 0:
        input_dim = sample_size - output_dim + input_dim
    preprocessor = preprocessors[mode]()
    return preprocessor.preprocess(table, input_dim, output_dim, **kwargs)
