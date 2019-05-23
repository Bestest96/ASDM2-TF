import tensorflow as tf
from models.Model import Model


class LayeredModel(Model):
    def __init__(self, build_params, default_params=None, x=None, name=None):
        self.layers = []
        super().__init__(build_params, default_params, x, name)

    def build(self, build_params):
        with tf.variable_scope(self.scope):
            self._build_layers(layers=self._prepare_layers(build_params))

    def _prepare_layers(self, build_params):
        raise NotImplementedError()

    def _build_layers(self, layers):
        for layer_type, params in layers:
            self._add_layer(layer_type, params)

    def _add_layer(self, layer_type, params):
        if len(self.layers) == 0:
            layer = layer_type(params, x=self.input_layer)
        else:
            layer = layer_type(params, x=self.layers[-1].layer)
        self.layers.append(layer)

    def output(self):
        return self.layers[-1].layer
