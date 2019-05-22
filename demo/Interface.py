import Resources as Resources


class ParameterInfo:
    def __init__(self, name, description='', error_description='', condition=None):
        self.name = name
        self.description = description
        self.error_description = error_description
        self.condition = condition

    def parse_integer(self, value):
        try:
            parsed_value = int(value)
            if self.condition is not None and self.condition(parsed_value):
                return parsed_value
            else:
                raise Exception(self.error_description)
        except Exception as e:
            raise Exception('Failed to convert' + self.name + ': ' + str(e))


class ExperimentsProperties:
    def __init__(self, problem_list, iteration_count, report_length, minibatch_length, run_count, algorithms, output_id):
        self.problem_list = problem_list
        self.iteration_count = iteration_count
        self.report_length = report_length
        self.minibatch_length = minibatch_length
        self.run_count = run_count
        self.algorithms = algorithms
        self.output_id = output_id


class Description:
    def __init__(self, dtype, parameters_dict=None):
        self.type = dtype
        self.parameters_dict = parameters_dict if parameters_dict is not None else dict()


def print_help():
    print("Usage: python main.py \"[problemDescriptionList] [iterationCount] [reportLength] "
          "[minibatchLength] [runCount] [testRunDescriptionList] [ID]\" [arguments]\n\n"
          "problemDescriptionList: [problemDescription1,problemDescription2,...]\t\t\n"
          "\tproblemDescription: [problemName]\n"
          "iterationCount:\t\tinteger > 0 number of teaching algorithm iterations\n"
          "reportLength:\t\tinteger > 0 number of samples after which reporting happens\n"
          "minibatchLength:\tinteger > 0 the size of a minibatch\n"
          "runCount:\t\tinteger > 0 number of runs of one experiment - results will be averaged\n"
          "testRunDescriptionList:\t[testRunDescription1,testRunDescription2,...]\n"
          "\ttestRunDescription: [algorithmName](parameterName1=value1,parameterName2=value2,...). "
          "(...) part is optional, not specified parameters will be set to default values\n"
          "ID:\t\t\tfile identifier\n\n"
          "Optional arguments:\n"
          "\t\t--allow-growth\t\tTell TensorFlow to take graphics card memory as needed (if using a GPU)\n"
          "\t\t--save-graph\t\tSave computation graph that can be later displayed in TensorBoard\n"
          "\t\t--alg-first\t\t\tIterate over problems algorithms first\n")


def parse_problem_list(problems_str):
    problem_str_list = problems_str.split(',')
    if len(problem_str_list) == 1 and problem_str_list[0] == 'ALL':
        return [Description(name) for name in Resources.Available_Problems.values]
    return parse_description_list(problems_str, Resources.Available_Problems)


def parse_algorithm_list(test_run_list):
    return parse_description_list(test_run_list, Resources.Available_Policies)


def parse_description_list(list_str, resource, separator=','):
    descriptions = list_str.split(separator)
    return [parse_description(description, resource) for description in descriptions]


def check_resource_name(name, resource):
    if name not in resource.res_dict:
        raise Exception(resource.res_name + " '" + name + "' not found.")


def parse_description(description, resource):
    if description.find('(') == -1:
        check_resource_name(description, resource)
        return Description(resource.res_dict[description])
    if description.find(')') == -1:
        raise Exception(resource.res_name + " description parsing failed!")
    name = description[:description.find('(')]

    check_resource_name(name, resource)

    parameters = description[description.find('(')+1:description.find(')')]
    try:
        params_dict = parse_parameters_list(parameters)
    except Exception as e:
        raise Exception(resource.res_name + " description parsing failed:\n" + str(e))

    return Description(resource.res_dict[name], params_dict)


def parse_parameters_list(parameters_str, separator=';'):
    assignments_list = parameters_str.split(separator)
    params_dict = dict()
    try:
        for assignment in assignments_list:
            param_name, param_value = assignment.split('=')
            params_dict[param_name] = float(param_value)
    except Exception as e:
        raise Exception("Parameter assignment parsing failed:\n" + str(e))
    return params_dict


def parse_program_input(argv):
    if len(argv) <= 6:
        print_help()
        raise Exception("Improper input. Too few arguments")

    iteration_count_param = ParameterInfo("iteration_count", error_description="is not an integer > 0",
                                          condition=lambda x: x > 0)
    report_length_param = ParameterInfo("report_length", error_description="is not an integer > 0",
                                       condition=lambda x: x > 0)
    minibatch_length_param = ParameterInfo("minibatch_length", error_description="is not an integer > 0",
                                           condition=lambda x: x > 0)
    run_count_param = ParameterInfo("run_count_param", error_description="is not an integer > 0",
                                    condition=lambda x: x > 0)

    problem_list = parse_problem_list(argv[0])

    iteration_count = iteration_count_param.parse_integer(argv[1])
    report_length = report_length_param.parse_integer(argv[2])
    minibatch_length = minibatch_length_param.parse_integer(argv[3])
    run_count = run_count_param.parse_integer(argv[4])

    algorithms = parse_algorithm_list(argv[5])

    output_file_id = argv[6]

    return ExperimentsProperties(problem_list,
                                 iteration_count, report_length, minibatch_length, run_count,
                                 algorithms,
                                 output_file_id)


def parse_arguments(argv):
    return set([arg[2:] for arg in argv if arg.startswith('--')])
