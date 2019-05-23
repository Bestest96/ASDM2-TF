import tensorflow as tf


def z_score_normalization_op(features):
    mean_v, var_v = tf.nn.moments(features, axes=0)
    return tf.nn.batch_normalization(features, mean=mean_v, variance=var_v, variance_epsilon=0.001, offset=0.0, scale=1.0)


class ScaleNormalization:

    def __init__(self, factor):
        self.factor = factor

    def operator(self, features):
        return features * self.factor


def scale_normalization_op(factor):
    scale_normalizator = ScaleNormalization(factor)
    return scale_normalizator.operator

