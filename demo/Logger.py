import traceback


class Logger:
    def __init__(self, filename, console=True):
        self.filename = filename
        self.console = console
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, "w")
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        self.file.close()

    def log_message(self, message):
        if self.console:
            print(message)
        self.file.write(message+'\n')

    def log_plot(self):
        pass


def log_run(logger, run, max_runs, problem, policy):
    logger.log_message('Run {} of {}'.format(run, max_runs))
    logger.log_message('Testing of {} started...'.format(problem.name))
    logger.log_message('Teaching policy: {}'.format(policy.name))
    log_parameters(logger, policy.parameters)


def log_parameters(logger, parameters):
    for param in parameters.items():
        logger.log_message('Explicit parameter: "{}" = {}'.format(param[0], param[1], 'e', 'e'))


def log_report(logger, iteration, loss, errors=None, algorithm_params=None):
    loss_str = str(loss) + '\t'
    if isinstance(errors, list):
        errors_str = "\t".join(str(error) for error in errors) + "\t" if errors is not None else ""
    else:
        errors_str = str(errors) + '\t'
    params_str = "\t".join(str(param) for param in algorithm_params) + "\t" if algorithm_params is not None else ""
    message = str(iteration) + "\t" + loss_str + errors_str + params_str
    logger.log_message(message)


def log_nan_error(logger):
    logger.log_message('Computation executing terminated - NaN occurred')
