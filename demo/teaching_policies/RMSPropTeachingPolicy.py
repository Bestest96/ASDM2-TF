import tensorflow as tf

from teaching_policies.TeachingPolicy import TeachingPolicy


class RMSPropTeachingPolicy(TeachingPolicy):

    default_parameters = {
        'learning_rate': 0.001,
        'decay': 0.9
    }

    name = 'RMSProp'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self, parameters_dict, RMSPropTeachingPolicy.default_parameters)

    def optimizer(self, loss, **kwargs):
        return tf.train.RMSPropOptimizer(learning_rate=self.parameters['learning_rate'],
                                         decay=self.parameters['decay']).minimize(loss)

    def get_info(self):
        return self.parameters.values()
