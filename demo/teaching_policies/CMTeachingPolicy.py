import tensorflow as tf

from teaching_policies.TeachingPolicy import TeachingPolicy


class CMTeachingPolicy(TeachingPolicy):

    default_parameters = {
        'learning_rate': 0.0005,
        'momentum': 0.99
    }

    name = 'CM'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self, parameters_dict, CMTeachingPolicy.default_parameters)

    def optimizer(self, loss, **kwargs):
        return tf.train.MomentumOptimizer(learning_rate=self.parameters['learning_rate'],
                                          momentum=self.parameters['momentum'],
                                          name="CM",
                                          use_nesterov=False).minimize(loss)

    def get_info(self):
        return self.parameters.values()
