import sys

import Interface as Interface
import TestingEnvironment as Env
from Logger import Logger


def main(argv):
    try:
        experiment_properties = Interface.parse_program_input(argv[1].split())
        arguments = Interface.parse_arguments(argv[2:])
    except Exception as e:
        print(e)
        return

    with Logger("test_" + experiment_properties.output_id + ".log", True) as logger:
        Env.run_experiments(experiment_properties, arguments, logger)


if __name__ == "__main__":
    main(sys.argv)
