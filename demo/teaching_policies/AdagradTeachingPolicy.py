import tensorflow as tf

from teaching_policies.TeachingPolicy import TeachingPolicy


class AdagradTeachingPolicy(TeachingPolicy):

    default_parameters = {
        'learning_rate': 0.02
    }

    name = 'Adagrad'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self, parameters_dict, AdagradTeachingPolicy.default_parameters)

    def optimizer(self, loss, **kwargs):
        return tf.train.AdagradOptimizer(learning_rate=self.parameters['learning_rate'],
                                         initial_accumulator_value=1.e-8).minimize(loss)

    def get_info(self):
        return self.parameters.values()
