import tensorflow as tf

from teaching_policies.TeachingPolicy import TeachingPolicy


class AdamTeachingPolicy(TeachingPolicy):

    default_parameters = {
        'learning_rate': 0.001,
        'beta1': 0.9,
        'beta2': 0.999
    }

    name = 'Adam'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self, parameters_dict, AdamTeachingPolicy.default_parameters)

    def optimizer(self, loss, **kwargs):
        return tf.train.AdamOptimizer(learning_rate=self.parameters['learning_rate'],
                                      beta1=self.parameters['beta1'],
                                      beta2=self.parameters['beta2']).minimize(loss)

    def get_info(self):
        return self.parameters.values()
