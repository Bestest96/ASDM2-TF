import tensorflow as tf

from teaching_policies.TeachingPolicy import TeachingPolicy


class AdadeltaTeachingPolicy(TeachingPolicy):

    default_parameters = {
        'learning_rate': 1,
        'rho': 0.9
    }

    name = 'Adadelta'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self, parameters_dict, AdadeltaTeachingPolicy.default_parameters)

    def optimizer(self, loss, **kwargs):
        return tf.train.AdadeltaOptimizer(learning_rate=self.parameters['learning_rate'],
                                          rho=self.parameters['rho']).minimize(loss)

    def get_info(self):
        return self.parameters.values()
