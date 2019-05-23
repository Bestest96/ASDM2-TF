import tensorflow as tf
from models.DenseLayer import DenseLayer
from models.Model import ModelParams, ModelBuildingError
from models.LayeredModel import LayeredModel
import numpy as np


class PerceptronParams(ModelParams):
    def __init__(self,
                 hidden_layers,
                 input_size=None,
                 output_size=None):
        super().__init__(input_size, output_size)
        self.hidden_layers_params = hidden_layers


class MLPerceptron(LayeredModel):
    """Multi layer perceptron model

    hidden_weights_init, hidden_biases_init and hidden_activation params
    concerns every hidden layer exclusive of layers defined in special_layers.

    If init_weights_and_biases_conjunctionally is True, hidden layer's weights as well as biases
    will be initialized by hidden_weights_init param. hidden_biases_init will be ignored.

    """
    default = {
        'hidden_weights_init': tf.variance_scaling_initializer(scale=6.25),
        'hidden_biases_init': tf.random_normal_initializer,
        'hidden_activation': tf.sigmoid,
        'output_weights_init': tf.zeros_initializer,
        'output_biases_init': tf.zeros_initializer,
        'output_activation': tf.identity,
        'init_weights_and_biases_conjunctionally': False,
        'special_layers': None  # dictionary with indices and params for special dense layers
    }

    def __init__(self, build_params: ModelParams, x=None, name=None):
        super().__init__(build_params, x=x, default_params=self.default, name=name)

    def _prepare_layers(self, build_params):

        # get hidden layers params
        layers_sizes = build_params.params.get('hidden_layers')
        hidden_biases_init = build_params.params.get('hidden_biases_init')
        hidden_weights_init = build_params.params.get('hidden_weights_init')
        hidden_activation = build_params.params.get('hidden_activation')
        init_wb_con = build_params.params.get('init_weights_and_biases_conjunctionally')
        special_layers = build_params.params.get('special_layers')

        # build hidden layers
        layers = []
        for size, i in zip(layers_sizes, range(len(layers_sizes))):
            if special_layers is None or special_layers.get(i) is None:
                params = ModelParams(
                    output_size=size,
                    weights_init=hidden_weights_init,
                    biases_init=hidden_biases_init,
                    activation=hidden_activation,
                    init_weights_and_biases_conjunctionally=init_wb_con
                )
            else:
                params = special_layers[i]
            layers.append((DenseLayer, params))

        # get output layer params
        out_b_init = build_params.params.get('output_biases_init')
        out_w_init = build_params.params.get('output_weights_init')
        out_activation = build_params.params.get('output_activation')

        # build output layer
        output_params = ModelParams(output_size=build_params.output_size[0],
                                    weights_init=out_w_init,
                                    bias_init=out_b_init,
                                    activation=out_activation,
                                    weight_bias_as_one_tensor=init_wb_con)

        return layers + [(DenseLayer, output_params)]

    @staticmethod
    def valid_params(build_params):
        layers_sizes = build_params.params.get('hidden_layers')

        if layers_sizes is None or len(layers_sizes) == 0:
            raise ModelBuildingError("Empty layers sizes list", build_params)

        special_layers = build_params.params.get('special_layers')

        if special_layers is not None and not isinstance(special_layers, dict):
            raise ModelBuildingError("The special_layers must be None or dictionary", build_params)

        input_rank = len(build_params.input_size)
        output_rank = len(build_params.output_size)

        if input_rank != 1:
            raise ModelBuildingError("Improper input rank. Input has rank {}, {} expected".format(input_rank, 1),
                                     build_params)
        if output_rank != 1:
            raise ModelBuildingError("Improper output rank. Output has rank {}, {} expected".format(input_rank, 1),
                                     build_params)


def one_layer_perceptron_params(hidden_neurons):
    """Based on given size of hidden layer returns params for default perceptron"""
    return ModelParams(
        hidden_layers=[hidden_neurons]
    )


def autoencoder_params(hidden_layers_list, **kwargs):
    """Based on given list creates multilayer perceptron with sigmoid activation
    for hidden layers and linear activation for the narrowest layer"""
    narrowest_idx = np.argmin(hidden_layers_list)
    narrowest_size = hidden_layers_list[narrowest_idx]
    return ModelParams(
        hidden_layers=hidden_layers_list,
        special_layers=dict({
            narrowest_idx: ModelParams(
                output_size=narrowest_size,
                activation=tf.identity
            )
        }),
        **kwargs
    )
