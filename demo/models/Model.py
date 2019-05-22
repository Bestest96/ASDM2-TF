import tensorflow as tf


class ModelParams:
    def __init__(self, input_size=None, output_size=None, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.params = kwargs

    def set_defaults(self, defaults: dict):
        self.params = {**defaults, **self.params}


class Model:
    def __init__(self, build_params, default_params=None, x=None, name=None):
        self.input_layer = None
        self.name = 'model' if name is None else name
        self.scope = tf.VariableScope(self.name)
        if default_params is not None:
            build_params.set_defaults(default_params)
        self.valid_params(build_params)
        self.prepare_input(build_params, x)
        self.build(build_params)

    def prepare_input(self, build_params, x=None):
        if x is None:
            if build_params.input_size is None:
                raise ModelBuildingError("Neither input tensor nor input size are defined", build_params)
            with tf.variable_scope(self.scope):
                self.input_layer = tf.placeholder(dtype='float', shape=[None] + build_params.input_size)
        else:
            self.input_layer = x

    def build(self, build_params):
        raise NotImplementedError()

    @staticmethod
    def valid_params(build_params):
        pass

    def input(self):
        return self.input_layer

    def output(self):
        raise NotImplementedError()


class ModelBuildingError(Exception):
    def __init__(self, message, model_params):
        super().__init__(self)
