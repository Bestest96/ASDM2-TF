"""ASDM2 Optimizer for TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.contrib.graph_editor as graph_editor
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class ASDM2Optimizer(optimizer.Optimizer):
    """
    Optimizer implementing the ASDM2 algorithm.
    """
    def __init__(self,
                 beta=1.0,
                 t0=10.0,
                 delta=0.0005,
                 c=1.0e8,
                 lambda_min=0.5,
                 mu_min=0.5,
                 mu_max=0.9999,
                 eps=1.0e-8,
                 use_grad_scaling=False,
                 rho=0.999,
                 use_ag=False,
                 use_locking=False,
                 name="ASDM2"):
        """
        Constructs a new ASDM2 optimizer object.

        Default values are taken from the paper describing the algorithm.
        :param beta: Initial value of the step size parameter.
        :param t0: Value representing how long initial learning period will last.
        :param delta: Value used to adjust logarithmic representations of step size,
                      momentum decay factor and mu values.
        :param c: Value dependant on representation of numbers defining how many times can one value be larger
                  from another with their addition still being effective.
        :param lambda_min: A minimum value of momentum decay factor.
        :param mu_min: A minimum value of mu parameter.
        :param mu_max: A maximum value of mu parameter.
        :param eps: Value preventing from division by zero in gradient scaling.
        :param use_grad_scaling: If True, gradient scaling will be used.
        :param rho: Exponential smoothing decay value.
        :param use_ag: If True, Accelerated Gradient version will be used, else Classic Momentum will be used
        :param use_locking: If True, use locks for update operations
        :param name: Optional name.
        """
        super(ASDM2Optimizer, self).__init__(use_locking, name)

        self._beta = beta
        self._lambda = 0.99
        self._gamma = 0.99
        self._mu = 0.99
        self._t0 = t0
        self._delta = delta
        self._C = c
        self._lambda_min = lambda_min
        self._mu_min = mu_min
        self._mu_max = mu_max
        self._eps = eps
        self._use_grad_scaling = use_grad_scaling
        self._rho = rho
        self._use_ag = use_ag

        self._t0_t = None
        self._delta_t = None
        self._C_t = None
        self._lambda_min_t = None
        self._mu_min_t = None
        self._mu_max_t = None
        self._rho_t = None
        if self._use_grad_scaling:
            self._eps_t = None

        self._grads = []
        self._vars = []
        self._loss = None
        self._theta_phi_loss = None

    """
    Prepares tensors from immutable algorithm parameters. 
    """
    def _prepare(self):
        self._t0_t = ops.convert_to_tensor(self._t0)
        self._delta_t = ops.convert_to_tensor(self._delta)
        self._C_t = ops.convert_to_tensor(self._C)
        self._lambda_min_t = ops.convert_to_tensor(self._lambda_min)
        self._mu_min_t = ops.convert_to_tensor(self._mu_min)
        self._mu_max_t = ops.convert_to_tensor(self._mu_max)
        self._rho_t = ops.convert_to_tensor(self._rho)
        if self._use_grad_scaling:
            self._eps_t = ops.convert_to_tensor(self._eps)

    """
    Creates slots for optimizer variables, colocates non slot variables with first variable. 
    """
    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        alpha = float(np.log(self._beta))
        eta = float(-np.log(1.0 - self._lambda))
        nu = float(-np.log(1.0 - self._mu))
        self._create_non_slot_variable(initial_value=self._beta,
                                       name="beta",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._lambda,
                                       name="lambda",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._gamma,
                                       name="gamma",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._mu,
                                       name="mu",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=alpha,
                                       name="alpha",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=eta,
                                       name="eta",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=nu,
                                       name="nu",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0.0,
                                       name="e_dq_db2",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0.0,
                                       name="e_dq_dl2",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0.0,
                                       name="e_dbj_dmu2",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0.0,
                                       name="e_g2",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0.0,
                                       name="e_m2",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=1.0,
                                       name="t",
                                       colocate_with=first_var)

        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "s_dm_db", self._name)
            self._zeros_slot(v, "s_dm_dl", self._name)
            self._zeros_slot(v, "s_dt_db", self._name)
            self._zeros_slot(v, "s_dt_dl", self._name)
            self._zeros_slot(v, "s_dbt_db", self._name)
            self._zeros_slot(v, "s_dbt_dl", self._name)
            self._zeros_slot(v, "dbt_dmu", self._name)
            self._zeros_slot(v, "phi", self._name)

            ones = 1.0
            for dim in reversed(v.shape.dims):
                ones_list = []
                for _ in range(dim):
                    ones_list.append(ones)
                ones = ones_list
            self._get_or_make_slot_with_initializer(v, ones, v.shape, dtypes.float32, "scaler", self._name)
            if self._use_grad_scaling:
                self._get_or_make_slot_with_initializer(v, ones, v.shape, dtypes.float32, "av_g2", self._name)

    """
    Intercepts vars and grads as updates are made in :_finish() method. 
    """
    def _apply_dense(self, grad, var):
        self._grads.append(grad)
        self._vars.append(var)

    """
    Intercepts vars and grads as updates are made in _finish() method. 
    """
    def _apply_sparse(self, grad, var):
        self._grads.append(grad)
        self._vars.append(var)

    """
    Not implemented
    """
    def _resource_apply_dense(self, grad, handle):
        pass

    """
    Not implemented
    """
    def _resource_apply_sparse(self, grad, handle, indices):
        pass

    """
    Intercepts loss graph to use in _finish() method.
    :return: minimize() of Optimizer base class. 
    """
    def minimize(self, loss, global_step=None, var_list=None, gate_gradients=optimizer.Optimizer.GATE_OP,
                 aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
        self._loss = loss
        return super().minimize(loss, global_step, var_list, gate_gradients, aggregation_method,
                                colocate_gradients_with_ops, name, grad_loss)

    """
    :return: Current graph.
    """
    def _get_graph(self):
        if context.in_eager_mode():
            return None
        return ops.get_default_graph()

    """
    Duplicates loss graph with swapped variables.
    :return: Swapped graph. 
    """
    def _duplicate_graph(self, graph, vars_to_replace, name='Duplicated'):
        if graph in vars_to_replace:
            return vars_to_replace[graph]

        operations = []

        def get_ops(t):
            if t.op.type != 'VariableV2' and t.op.type != 'Placeholder':
                operations.append(t.op)
                for i in t.op.inputs:
                    if i not in vars_to_replace:
                        get_ops(i)

        get_ops(graph)

        sgv = graph_editor.make_view(operations)
        with ops.name_scope(name):
            new_view, _ = graph_editor.copy_with_input_replacements(sgv, vars_to_replace)
            return new_view.outputs[sgv.output_index(graph)]

    """
    Calculates gradients for swapped variables.
    :return: Gradient values for swapped variables and new loss graph from _duplicate_graph()
    """
    def _get_gradient_other_vars(self, new_variables, loss):
        new_vars_dict = dict(zip([v.op.outputs[0] for v in self._vars], new_variables))
        new_loss = self._duplicate_graph(loss, new_vars_dict)
        return gradients.gradients(xs=new_variables, ys=new_loss, name="grad2"), new_loss

    """
    Updates gradient scaler value.
    :return: Assignments of new scaler values and new scaler value. 
    """
    def _update_scaler(self, decay, graph):
        assignments = []
        scaler_values = []
        for v, g in zip(self._vars, self._grads):
            av_g2 = decay * self.get_slot(v, "av_g2") + (array_ops.constant(1.0) - decay) * math_ops.square(g)
            scaler = math_ops.rsqrt(av_g2 + self._eps_t)
            scaler_values.append(scaler)
            assignments.append(state_ops.assign(self.get_slot(v, "av_g2"), av_g2))
            assignments.append(state_ops.assign(self.get_slot(v, "scaler"), scaler))
        return assignments, scaler_values

    """
    Scales values with scaler.
    :return: Scaled values. 
    """
    def _scale_values(self, values, scaler):
        return [v * s for v, s in zip(values, scaler)]
    """
    Calculates a hessian-vector product estimation with gradient calculated earlier. 
    :return: An estimate of hessian-vector product. 
    """
    def _hvp(self, grad, var, vec):
        return gradients.gradients([g * array_ops.stop_gradient(v) for g, v in zip(grad, vec)], [v for v in var])

    """
    Caclulates dQ/dB, dQ/dL and dJ/dmu sums. 
    :return: Values of calculated sums. 
    """
    def _get_sums(self, scaled_gtp):
        dq_db = math_ops.add_n([math_ops.reduce_sum(gtp * (self.get_slot(v, "s_dbt_db") + self.get_slot(v, "s_dm_db")))
                                for gtp, v in zip(scaled_gtp, self._vars)])
        dq_dl = math_ops.add_n([math_ops.reduce_sum(gtp * (self.get_slot(v, "s_dbt_dl") + self.get_slot(v, "s_dm_dl")))
                                for gtp, v in zip(scaled_gtp, self._vars)])
        dbj_dmu = math_ops.add_n(
            [math_ops.reduce_sum(gtp * self.get_slot(v, "dbt_dmu")) for gtp, v in zip(scaled_gtp, self._vars)])
        return dq_db, dq_dl, dbj_dmu
    """
    Calculates values for t <= t0 condition part.
    :return: Calculated values that are later needed in calculations.
    """
    def _calculate_t0_cond_values(self, scaled_g, scaler, graph):
        t = self._get_non_slot_variable("t", graph)
        cond_t0 = math_ops.less_equal(t, self._t0_t)
        grad_norm = math_ops.sqrt(math_ops.add_n([math_ops.reduce_sum(math_ops.square(g)) for g in scaled_g]))
        beta_val = self._get_non_slot_variable("beta", graph)
        beta = control_flow_ops.cond(cond_t0,
                                     lambda: array_ops.constant(0.5) / (array_ops.constant(0.5) / beta_val *
                                                                        (t - array_ops.constant(1.0)) / t +
                                                                        math_ops.sqrt(math_ops.add_n(
                                                                            [math_ops.reduce_sum(
                                                                                math_ops.square(hvp_s))
                                                                                for hvp_s in self._scale_values(
                                                                                self._hvp(self._grads, self._vars,
                                                                                          scaled_g),
                                                                                scaler)]))
                                                                        / grad_norm / t),
                                     lambda: beta_val)
        alpha = control_flow_ops.cond(cond_t0,
                                      lambda: math_ops.log(beta),
                                      lambda: self._get_non_slot_variable("alpha", graph))
        lambd = control_flow_ops.cond(cond_t0,
                                      lambda: array_ops.constant(0.5),
                                      lambda: self._get_non_slot_variable("lambda", graph))
        eta = control_flow_ops.cond(cond_t0,
                                    lambda: -math_ops.log(array_ops.constant(1.0) - lambd),
                                    lambda: self._get_non_slot_variable("eta", graph))
        return alpha, beta, eta, lambd

    """
    Calculates values for t > t0 condition part.
    :return: Assignments made and values that are later needed in calculations.
    """
    def _calculate_neg_t0_cond_values(self, dq_db, dq_dl, dbj_dmu, alpha_val, beta_val, eta_val, lambda_val, graph):
        assignments = []
        t = self._get_non_slot_variable("t", graph)
        cond_neg_t0 = math_ops.greater(t, self._t0_t)
        alpha = control_flow_ops.cond(cond_neg_t0,
                                      lambda: alpha_val - self._delta_t * dq_db / math_ops.sqrt(
                                          self._get_non_slot_variable("e_dq_db2", graph)),
                                      lambda: alpha_val)
        beta = control_flow_ops.cond(cond_neg_t0,
                                     lambda: math_ops.exp(alpha),
                                     lambda: beta_val)
        eta = control_flow_ops.cond(cond_neg_t0,
                                    lambda: math_ops.maximum(eta_val - self._delta_t * dq_dl /
                                                             math_ops.sqrt(
                                                                 self._get_non_slot_variable("e_dq_dl2", graph)),
                                                             -math_ops.log(
                                                                 array_ops.constant(1.0) - self._lambda_min_t)),
                                    lambda: eta_val)
        lambd = control_flow_ops.cond(cond_neg_t0,
                                      lambda: array_ops.constant(1.0) - math_ops.exp(-eta),
                                      lambda: lambda_val)
        nu_val = self._get_non_slot_variable("nu", graph)
        mu_val = self._get_non_slot_variable("mu", graph)
        nu = control_flow_ops.cond(cond_neg_t0,
                                   lambda: math_ops.minimum(math_ops.maximum(
                                       -math_ops.log(array_ops.constant(1.0) - self._mu_min_t),
                                       nu_val - self._delta_t * dbj_dmu
                                       / math_ops.sqrt(self._get_non_slot_variable("e_dbj_dmu2", graph))),
                                       -math_ops.log(array_ops.constant(1.0) - self._mu_max_t)),
                                   lambda: nu_val)
        assignments.append(state_ops.assign(self._get_non_slot_variable("nu", graph), nu))
        mu = control_flow_ops.cond(cond_neg_t0,
                                   lambda: array_ops.constant(1.0) - math_ops.exp(-nu),
                                   lambda: mu_val)
        assignments.append(state_ops.assign(self._get_non_slot_variable("mu", graph), mu))
        return assignments, alpha, beta, eta, lambd, mu, nu

    """
    Calculates values for t > 1 condition part.
    :return: Assignments made and values that are later needed in calculations.
    """
    def _calculate_t1_cond_values(self, alpha_val, beta_val, eta_val, lambda_val, scaled_s_dg_db, scaled_s_dg_dl,
                                  graph):
        assignments = []
        t = self._get_non_slot_variable("t", graph)
        cond_t1 = math_ops.greater(t, array_ops.constant(1.0))
        s_dt_db_norm = math_ops.sqrt(math_ops.add_n([math_ops.reduce_sum(math_ops.square(self.get_slot(v, "s_dt_db")))
                                                     for v in self._vars]))
        s_dt_dl_norm = math_ops.sqrt(math_ops.add_n([math_ops.reduce_sum(math_ops.square(self.get_slot(v, "s_dt_dl")))
                                                     for v in self._vars]))
        alpha0 = control_flow_ops.cond(cond_t1,
                                       lambda: math_ops.log(array_ops.constant(0.5) *
                                                            math_ops.minimum(s_dt_db_norm /
                                                                             math_ops.sqrt(
                                                                                 math_ops.add_n(
                                                                                     [math_ops.reduce_sum(
                                                                                         math_ops.square(sgb))
                                                                                         for sgb in scaled_s_dg_db])),
                                                                             s_dt_dl_norm /
                                                                             math_ops.sqrt(
                                                                                 math_ops.add_n(
                                                                                     [math_ops.reduce_sum(
                                                                                         math_ops.square(sgl))
                                                                                         for sgl in scaled_s_dg_dl])))),
                                       lambda: array_ops.constant(float('Inf')))
        cond_a0 = math_ops.greater(alpha_val, alpha0)
        alpha = control_flow_ops.cond(math_ops.logical_and(cond_t1, cond_a0),
                                      lambda: alpha_val - array_ops.constant(2.0) * self._delta_t,
                                      lambda: alpha_val)
        beta = control_flow_ops.cond(math_ops.logical_and(cond_t1, cond_a0),
                                     lambda: math_ops.exp(alpha0),
                                     lambda: beta_val)
        eg2 = self._get_non_slot_variable("e_g2", graph)
        em2 = self._get_non_slot_variable("e_m2", graph)
        gamma = control_flow_ops.cond(cond_t1,
                                      lambda: math_ops.minimum(array_ops.constant(1.0),
                                                               math_ops.minimum(self._C_t * eg2 /
                                                                                math_ops.square(s_dt_db_norm),
                                                                                self._C_t * em2 /
                                                                                math_ops.square(s_dt_dl_norm))),
                                      lambda: self._get_non_slot_variable("gamma", graph))
        cond_gl = math_ops.greater(lambda_val, gamma)
        eta = control_flow_ops.cond(math_ops.logical_and(cond_t1, cond_gl),
                                    lambda: math_ops.maximum(-math_ops.log(array_ops.constant(1.0) - self._lambda_min_t)
                                                             , eta_val - array_ops.constant(2.0) * self._delta_t),
                                    lambda: eta_val)
        lambd = control_flow_ops.cond(math_ops.logical_and(cond_t1, cond_gl),
                                      lambda: array_ops.constant(1.0) - math_ops.exp(-eta),
                                      lambda: lambda_val)
        assignments.append(state_ops.assign(self._get_non_slot_variable("alpha", graph), alpha))
        assignments.append(state_ops.assign(self._get_non_slot_variable("beta", graph), beta))
        assignments.append(state_ops.assign(self._get_non_slot_variable("eta", graph), eta))
        assignments.append(state_ops.assign(self._get_non_slot_variable("lambda", graph), lambd))
        assignments.append(state_ops.assign(self._get_non_slot_variable("gamma", graph), gamma))
        return assignments, beta, lambd, gamma

    """
    Updates variables and estimator values.
    :return: Assignments of new values.
    """
    def _update_vars_and_estimators(self, scaled_g, beta, prev_lambd, lambd, gamma, mu, scaled_s_dg_db, scaled_s_dg_dl,
                                    dq_db, dq_dl, dbj_dmu, w_t, graph):
        assignments = []
        momentum_values = []
        for v, g, sgb, sgl in zip(self._vars, scaled_g, scaled_s_dg_db, scaled_s_dg_dl):
            sbtb = self.get_slot(v, "s_dbt_db")
            sbtl = self.get_slot(v, "s_dbt_dl")
            p = self.get_slot(v, "phi")
            btmu = self.get_slot(v, "dbt_dmu")
            stb = self.get_slot(v, "s_dt_db")
            stl = self.get_slot(v, "s_dt_dl")
            smb = self.get_slot(v, "s_dm_db")
            sml = self.get_slot(v, "s_dm_dl")
            m = self.get_slot(v, "momentum")
            if not self._use_ag:
                s_dbt_db = mu * gamma * sbtb + (array_ops.constant(1.0) - mu) * gamma * stb
                s_dbt_dl = mu * gamma * sbtl + (array_ops.constant(1.0) - mu) * gamma * stl
                dbt_dmu = -p + mu * btmu
                s_dm_db = lambd * gamma * smb - g - beta * gamma * sgb
                s_dm_dl = m + lambd * gamma * sml - beta * gamma * sgl
                momentum = lambd * m - beta * g
                new_v = v + momentum
                phi = mu * p + momentum
                s_dt_db = gamma * stb + s_dm_db
                s_dt_dl = gamma * stl + s_dm_dl
            else:
                s_dbt_db = mu * gamma * sbtb + \
                           (array_ops.constant(1.0) - mu) * gamma * (stb - lambd * smb)
                s_dbt_dl = mu * gamma * sbtl + \
                           (array_ops.constant(1.0) - mu) * (gamma * stl - m - lambd * gamma * sml)
                dbt_dmu = -p + mu * btmu
                s_dm_db = prev_lambd * gamma * smb - g - beta * gamma * sgb
                s_dm_dl = m + prev_lambd * gamma * sml - beta * sgl
                momentum = prev_lambd * m - beta * g
                new_v = v + lambd * momentum - beta * g
                phi = mu * p + momentum
                s_dt_db = gamma * stb + lambd * s_dm_db - g - beta * gamma * sgb
                s_dt_dl = gamma * stl + momentum + lambd * gamma * s_dm_dl - beta * gamma * sgl
            momentum_values.append(momentum)
            assignments.append(state_ops.assign(self.get_slot(v, "s_dbt_db"), s_dbt_db))
            assignments.append(state_ops.assign(self.get_slot(v, "s_dbt_dl"), s_dbt_dl))
            assignments.append(state_ops.assign(self.get_slot(v, "dbt_dmu"), dbt_dmu))
            assignments.append(state_ops.assign(self.get_slot(v, "s_dm_db"), s_dm_db))
            assignments.append(state_ops.assign(self.get_slot(v, "s_dm_dl"), s_dm_dl))
            assignments.append(state_ops.assign(self.get_slot(v, "momentum"), momentum))
            assignments.append(state_ops.assign(v, new_v))
            assignments.append(state_ops.assign(self.get_slot(v, "phi"), phi))
            assignments.append(state_ops.assign(self.get_slot(v, "s_dt_db"), s_dt_db))
            assignments.append(state_ops.assign(self.get_slot(v, "s_dt_dl"), s_dt_dl))
        e_dq_db2 = w_t * self._get_non_slot_variable("e_dq_db2", graph) + \
                   (array_ops.constant(1.0) - w_t) * math_ops.square(dq_db)
        e_dq_dl2 = w_t * self._get_non_slot_variable("e_dq_dl2", graph) + \
                   (array_ops.constant(1.0) - w_t) * math_ops.square(dq_dl)
        e_dbj_dmu2 = w_t * self._get_non_slot_variable("e_dbj_dmu2", graph) + \
                     (array_ops.constant(1.0) - w_t) * math_ops.square(dbj_dmu)
        e_g2 = w_t * self._get_non_slot_variable("e_g2", graph) + \
               (array_ops.constant(1.0) - w_t) * math_ops.add_n([math_ops.reduce_sum(math_ops.square(g))
                                                                 for g in scaled_g])
        e_m2 = w_t * self._get_non_slot_variable("e_m2", graph) + \
               (array_ops.constant(1.0) - w_t) * math_ops.add_n([math_ops.reduce_sum(math_ops.square(m))
                                                                 for m in momentum_values])
        assignments.append(state_ops.assign(self._get_non_slot_variable("e_dq_db2", graph), e_dq_db2))
        assignments.append(state_ops.assign(self._get_non_slot_variable("e_dq_dl2", graph), e_dq_dl2))
        assignments.append(state_ops.assign(self._get_non_slot_variable("e_dbj_dmu2", graph), e_dbj_dmu2))
        assignments.append(state_ops.assign(self._get_non_slot_variable("e_g2", graph), e_g2))
        assignments.append(state_ops.assign(self._get_non_slot_variable("e_m2", graph), e_m2))
        with ops.control_dependencies(assignments):
            assignments.append(state_ops.assign_add(self._get_non_slot_variable("t", graph), 1.0))

        return assignments


    """
    Does calculations defined in ASDM2 algorithm.
    :return: All assignments of an algorithm plus some values to be reported to the user. 
    """
    def _finish(self, _, name_scope):
        assignments = []
        graph = self._get_graph()
        prev_lambd = self._get_non_slot_variable("lambda", graph) + array_ops.constant(0.0)
        theta_phi_vars = [v - self.get_slot(v, "phi") if not self._use_ag
                          else v - (self.get_slot(v, "phi")
                                    + prev_lambd * self.get_slot(v, "momentum"))
                          for v in self._vars]
        t = self._get_non_slot_variable("t", graph)
        w_t = self._rho_t * \
              (array_ops.constant(1.0) - math_ops.pow(self._rho_t, t - array_ops.constant(1.0))) / \
              (array_ops.constant(1.0) - math_ops.pow(self._rho_t, t))
        if self._use_grad_scaling:
            scaler_assignments, scaler = self._update_scaler(w_t, graph)
            assignments.extend(scaler_assignments)
        else:
            scaler = [self.get_slot(v, "scaler") for v in self._vars]
        theta_phi_grad_ns, self._theta_phi_loss = self._get_gradient_other_vars(theta_phi_vars, self._loss)
        scaled_grad = self._scale_values(self._grads, scaler)
        scaled_theta_phi_grad = self._scale_values(theta_phi_grad_ns, scaler)
        dq_db, dq_dl, dbj_dmu = self._get_sums(scaled_theta_phi_grad)
        alpha, beta, eta, lambd = self._calculate_t0_cond_values(scaled_grad, scaler, graph)
        neg_t0_assignments, alpha, beta, eta, lambd, mu, nu = self._calculate_neg_t0_cond_values(
            dq_db, dq_dl, dbj_dmu, alpha, beta, eta, lambd, graph)
        assignments.extend(neg_t0_assignments)
        s_dg_db_ns = self._hvp(self._grads, self._vars,
                               [self.get_slot(v, "s_dt_db") for v in self._vars])
        s_dg_dl_ns = self._hvp(self._grads, self._vars,
                               [self.get_slot(v, "s_dt_dl") for v in self._vars])
        s_dg_db = self._scale_values(s_dg_db_ns, scaler)
        s_dg_dl = self._scale_values(s_dg_dl_ns, scaler)
        t1_assignments, beta, lambd, gamma = self._calculate_t1_cond_values(alpha, beta, eta, lambd, s_dg_db, s_dg_dl,
                                                                            graph)
        assignments.extend(t1_assignments)
        update_assignments = self._update_vars_and_estimators(scaled_grad, beta, prev_lambd, lambd, gamma, mu,
                                                              s_dg_db, s_dg_dl, dq_db, dq_dl, dbj_dmu, w_t, graph)
        assignments.extend(update_assignments)
        return control_flow_ops.group(assignments, name=name_scope), [beta, lambd, gamma,
                                                                      array_ops.constant(1.0) - math_ops.exp(-nu),
                                                                      self._loss, self._theta_phi_loss]
