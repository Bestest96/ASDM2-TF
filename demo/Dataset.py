from random import shuffle, sample

import numpy as np


class Dataset:
    def __init__(self, xs, ys):
        self.samples = [(x, y) for x, y in zip(xs, ys)]
        shuffle(self.samples)
        self.used_samples = 0

    def __len__(self):
        return len(self.samples)

    def input_size(self):
        return list(self.samples[0][0].shape)

    def output_size(self):
        return list(self.samples[0][1].shape)

    def class_count(self):
        return max(self.samples, key=lambda x: x[1])[1] + 1

    def get_minibatch(self, size):
        sample_indices = sample(range(len(self.samples)), size)
        minibatch_tuples = [self.samples[i] for i in sample_indices]
        sep = [list(s) for s in zip(*minibatch_tuples)]
        return np.array(sep[0]), np.reshape(sep[1], (size, *self.output_size()))
