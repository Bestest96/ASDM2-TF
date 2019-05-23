import tensorflow as tf


def error_opers_tuple(error_opers: list, output_t, desired_output_t):

    return tf.tuple(
        [operator(output_t, desired_output_t) for operator in error_opers]
    )


def prediction_error(output_t, desired_output_t):
    prediction = tf.argmax(output_t, 1)
    equality = tf.not_equal(prediction, tf.argmax(desired_output_t, 1))
    error = tf.reduce_mean(tf.cast(equality, tf.float32))
    return error


def mean_square_error(output_t, desired_output_t, name='MSE_loss'):
    with tf.name_scope(name):
        return tf.reduce_sum(tf.reduce_mean(tf.squared_difference(output_t, desired_output_t), axis=0))


def cross_entropy(output_t, desired_output_t, name='cross_entropy_loss'):
    sample_axis = list(range(1, len(output_t.shape)))
    out_log = tf.log(tf.maximum(output_t, 1e-12))
    inv_out_log = tf.log(tf.maximum(1.0 - output_t, 1e-12))
    inv_des_out = 1.0 - desired_output_t
    sample_cross_entropy = tf.reduce_sum(-desired_output_t * out_log - inv_des_out * inv_out_log, axis=sample_axis)
    return tf.reduce_mean(sample_cross_entropy, name=name)
