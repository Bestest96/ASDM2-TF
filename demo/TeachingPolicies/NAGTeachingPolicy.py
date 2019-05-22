from TeachingPolicies.TeachingPolicy import TeachingPolicy

import tensorflow as tf


class NAGTeachingPolicy(TeachingPolicy):

    default_parameters = {
        'learning_rate': 0.02,
        'momentum': 0.99
    }

    name = 'NAG'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self,
                                parameters_dict,
                                NAGTeachingPolicy.default_parameters)

    def optimizer(self, loss, **kwargs):
        return tf.train.MomentumOptimizer(learning_rate=self.parameters['learning_rate'],
                                          momentum=self.parameters['momentum'],
                                          name="NAG",
                                          use_nesterov=True).minimize(loss)

    def get_info(self):
        return self.parameters.values()
