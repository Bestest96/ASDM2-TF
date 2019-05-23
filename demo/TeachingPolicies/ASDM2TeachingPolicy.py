from TeachingPolicies.TeachingPolicy import TeachingPolicy
from TeachingPolicies.optimizers.asdm2 import ASDM2Optimizer


class ASDM2TeachingPolicy(TeachingPolicy):

    default_parameters = {
        "beta": 1.0,
        "t0": 10.0,
        "delta": 0.0005,
        "c": 1.0e8,
        "lambda_min": 0.5,
        "mu_min": 0.5,
        "mu_max": 0.9999,
        "eps": 1e-8,
        "rho": 0.999,
        "use_ag": 0,
        "use_grad_scaling": 0
    }

    name = 'ASDM2'

    def __init__(self, parameters_dict):
        TeachingPolicy.__init__(self, parameters_dict, ASDM2TeachingPolicy.default_parameters)
        self.lr = None
        self.mu = None
        self.gamma = None
        self.nu = None
        self.loss = None
        self.theta_phi_loss = None

    def optimizer(self, loss, **kwargs):
        return ASDM2Optimizer(beta=self.parameters["beta"],
                              t0=self.parameters["t0"],
                              delta=self.parameters["delta"],
                              c=self.parameters["c"],
                              lambda_min=self.parameters["lambda_min"],
                              mu_min=self.parameters["mu_min"],
                              mu_max=self.parameters["mu_max"],
                              eps=self.parameters["eps"],
                              use_grad_scaling=bool(self.parameters["use_grad_scaling"]),
                              rho=self.parameters["rho"],
                              use_ag=bool(self.parameters["use_ag"])).minimize(loss)

    def get_info(self):
        return [self.lr, self.mu, self.gamma, self.nu, self.loss, self.theta_phi_loss]
