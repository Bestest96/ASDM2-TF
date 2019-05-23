from models.Model import Model, ModelParams


class Layer(Model):

    def __init__(self, params: ModelParams, default_params=None, x=None, name=None):
        self.layer = None
        super().__init__(params, default_params, x, name=name)

    def build(self, build_params):
        raise NotImplementedError()

    def output(self):
        return self.layer
