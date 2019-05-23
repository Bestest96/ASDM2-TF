def init_parameters(defaults, given):
    params = dict(defaults)
    for key in given:
        if key in defaults:
            params[key] = given[key]
    return params


class TeachingPolicy:
    name = ''

    def __init__(self, parameters_dict, default_parameters):
        self.parameters = init_parameters(default_parameters, parameters_dict)

    def optimizer(self, loss, **kwargs):
        raise NotImplementedError()

    def update_mutable_params(self, params):
        pass

    def get_info(self):
        return self.parameters.values()
