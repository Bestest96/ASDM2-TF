import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + tf.exp(-tf.minimum(x, 80)))
