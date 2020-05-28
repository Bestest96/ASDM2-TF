import numpy as np
import tensorflow as tf

import Logger as Logger


class Experiment:
    def __init__(self, problem_desc, algorithm_desc):
        try:
            self.problem = problem_desc.type(problem_desc.parameters_dict)
        except FileNotFoundError:
            raise ProblemInitializationError("File not found", problem_desc.type)
        self.algorithm = algorithm_desc.type(algorithm_desc.parameters_dict)

    def run(self, iteration_count, report_length, minibatch_length, flags, logger=None):
        dataset = self.problem.dataset
        model = self.problem.build_model()
        labels = tf.placeholder(tf.float32, model.output().shape, name='label')
        error_op = self.problem.get_error_operators_tuple(model.output(), labels)
        loss = self.problem.loss_oper(model.output(), labels)
        optimizer = self.algorithm.optimizer(loss, minibatch_size=minibatch_length)

        save_graph = 'save-graph' in flags
        gpu_allow_growth = 'allow-growth' in flags

        report_count = parts_count(iteration_count, report_length)
        minibatch_count = parts_count(report_length, minibatch_length)

        init_op = tf.global_variables_initializer()
        training_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        config = tf.ConfigProto()
        if gpu_allow_growth:
            config.gpu_options.allow_growth = True

        errors_log = []
        losses_log = []
        opt_values_log = []

        with tf.Session(config=config) as sess:

            writer = None

            if save_graph:
                writer = tf.summary.FileWriter('graph', sess.graph)

            sess.run(init_op)

            for report in range(report_count):
                minibatch_errors_log = []
                minibatch_losses_log = []
                minibatch_opt_values_log = []
                l = None
                for _ in range(minibatch_count):
                    minibatch = dataset.get_minibatch(minibatch_length)
                    minibatch_dict = {model.input(): minibatch[0], labels: minibatch[1]}
                    res, minibatch_error, l, opt, trainable = sess.run([model.output(), error_op, loss,
                                                                        optimizer, training_variables], minibatch_dict)
                    minibatch_errors_log.append(minibatch_error)
                    minibatch_losses_log.append(l)
                    if opt is not None and len(opt) > 1:
                        minibatch_opt_values_log.append(opt[1])

                # check if NaN error
                if np.isnan(l).any():
                    Logger.log_nan_error(logger)
                    break

                report_errors = np.mean(minibatch_errors_log, axis=0)
                errors_log.append(report_errors)
                report_loss = sum(minibatch_losses_log) / minibatch_count
                losses_log.append(report_loss)
                if len(minibatch_opt_values_log) > 0:
                    opt_values = np.mean(minibatch_opt_values_log, axis=0)
                else:
                    opt_values = list(self.algorithm.get_info())
                opt_values_log.append(opt_values)
                if logger is not None:
                    Logger.log_report(logger,
                                      (report + 1) * report_length,
                                      errors=report_errors,
                                      loss=report_loss,
                                      algorithm_params=opt_values)

            if writer is not None:
                writer.close()


def parts_count(sup_len, sub_len):
    return int(max(1, sup_len / sub_len))


def run_experiments(exp_properties, arguments, logger=None):
    problems = exp_properties.problem_list
    algorithms = exp_properties.algorithms
    algorithm_first = 'alg-first' in arguments
    if algorithm_first:
        pairs = [(problem, algorithm) for problem in problems for algorithm in algorithms]
    else:
        pairs = [(problem, algorithm) for algorithm in algorithms for problem in problems]

    experiments = []
    for pair in pairs:
        try:
            exp = Experiment(pair[0], pair[1])
            experiments.append(exp)
        except ProblemInitializationError as e:
            print(str(e))
            print("Experiment '{}' - '{}' ignored".format(pair[0].type.name, pair[1].type.name))

    for experiment in experiments:
        for i in range(exp_properties.run_count):
            if logger is not None:
                Logger.log_run(logger, i + 1, exp_properties.run_count,
                               experiment.problem, experiment.algorithm)
            with tf.device('/gpu:0'):
                experiment.run(exp_properties.iteration_count,
                               exp_properties.report_length,
                               exp_properties.minibatch_length,
                               arguments,
                               logger=logger)
                tf.reset_default_graph()


class ProblemInitializationError(Exception):
    def __init__(self, message, problem_type):
        msg = "Problem '{}' initialization error: {}".format(problem_type.name, message)
        super().__init__(self, msg)
