from models.Layer import ModelParams, Layer
import tensorflow as tf


class DenseLayer(Layer):

    default = {'activation': tf.nn.sigmoid,
               'weights_init': tf.variance_scaling_initializer(scale=6.25),
               'bias_init': tf.random_normal_initializer,
               'init_weights_and_biases_conjunctionally': False}

    def __init__(self, build_params, x=None):
        super().__init__(build_params, self.default, x, name='Dense')

    def build(self, build_params: ModelParams):
        output_size = build_params.output_size
        activation = build_params.params.get('activation')
        wb_as_one = build_params.params.get('init_weights_and_biases_conjunctionally')
        w_init = build_params.params.get('weights_init')
        b_init = build_params.params.get('bias_init')
        if wb_as_one:
            weights_nb = self.input_layer.shape[1].value
            wb = tf.Variable(w_init([weights_nb + 1, output_size]))
            w = tf.slice(wb, [0, 0], [weights_nb, output_size])
            b = tf.slice(wb, [weights_nb, 0], [1, output_size])
            self.layer = activation(tf.add(tf.matmul(self.input_layer, w), b))
        else:
            self.layer = tf.layers.dense(self.input_layer, output_size,
                                         activation=activation,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init)

    @staticmethod
    def valid_params(build_params):
        pass

    def shape(self):
        return self.layer.shape()
