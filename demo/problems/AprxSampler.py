import copy

from error_operators import error_opers_tuple, mean_square_error
from file_readers import read_file


class AprxSampler:
    def __init__(self, model_type, loss_oper, error_opers=None):
        self.model_type = model_type
        self.model_params = None
        self.loss_oper = loss_oper
        self.error_opers = mean_square_error if error_opers is None or len(error_opers) == 0 else error_opers

    def set_model_params(self):
        raise NotImplementedError()

    def get_minibatch(self, size):
        raise NotImplementedError()

    def build_model(self, x=None):
        if self.model_params.input_size is None or self.model_params.output_size is None:
            self.model_params = copy.copy(self.model_params)
            if self.model_params.input_size is None:
                self.model_params.input_size = self.problem_dim()[0]
            if self.model_params.output_size is None:
                self.model_params.output_size = self.problem_dim()[1]
        return self.model_type(self.model_params, x=x)

    def problem_dim(self):
        raise NotImplementedError()

    def get_error_operators_tuple(self, output_t, desired_output_t):
        if not isinstance(self.error_opers, list):
            return self.error_opers(output_t, desired_output_t)
        if len(self.error_opers) == 1:
            return self.error_opers[0](output_t, desired_output_t)
        return error_opers_tuple(self.error_opers, output_t, desired_output_t)


class AprxDataSetSampler(AprxSampler):
    def __init__(self,
                 file_name,
                 mode,
                 model_type,
                 loss_oper,
                 error_opers=None,
                 input_dim=None,
                 output_dim=None,
                 **kwargs):
        AprxSampler.__init__(self, model_type, loss_oper=loss_oper, error_opers=error_opers)
        self.dataset = read_file(file_name,
                                 mode=mode,
                                 input_dim=input_dim,
                                 output_dim=output_dim,
                                 **kwargs)
        self.model_type = model_type
        self.model_params = self.set_model_params()

    def set_model_params(self):
        raise NotImplementedError()

    def get_minibatch(self, size):
        return self.dataset.get_minibatch(size)

    def problem_dim(self):
        input_size = self.dataset.input_size()
        output_size = self.dataset.output_size()
        return input_size, output_size
