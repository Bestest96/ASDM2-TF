from Problems.AprxSampler import AprxDataSetSampler

from models.MLPerceptron import MLPerceptron
from models.MLPerceptron import one_layer_perceptron_params, autoencoder_params

import error_operators as errors
import activations as activations

from normalization import z_score_normalization_op, scale_normalization_op


class AprxCreditCardUci(AprxDataSetSampler):

    name = 'CreditCardUci'

    def __init__(self, parameters_dict):

        AprxDataSetSampler.__init__(self,
                                    file_name='CreditCard.uci',
                                    mode='classification',
                                    model_type=MLPerceptron,
                                    loss_oper=errors.mean_square_error,
                                    error_opers=[errors.prediction_error],
                                    normalization=z_score_normalization_op)
        self.parameters = parameters_dict

    def set_model_params(self):
        return one_layer_perceptron_params(21)


class AprxHandwrittenDigitsMnistBin(AprxDataSetSampler):

    name = 'HandwrittenDigitsMnist'

    def __init__(self, parameters_dict):
        AprxDataSetSampler.__init__(self,
                                    file_name='HandwrittenDigitsMnist.bin',
                                    mode='autoencoder',
                                    model_type=MLPerceptron,
                                    output_dim=10,
                                    loss_oper=errors.cross_entropy,
                                    normalization=scale_normalization_op(1/255),
                                    error_opers=[errors.mean_square_error])
        self.parameters = parameters_dict

    def set_model_params(self):
        return autoencoder_params([1000, 500, 250, 30, 250, 500, 1000],
                                  output_activation=activations.sigmoid)
