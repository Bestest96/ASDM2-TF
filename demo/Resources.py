from TeachingPolicies.TeachingPolicy import TeachingPolicy
from Problems.AprxSampler import AprxSampler


class Resource:
    def __init__(self, res_dict, res_name):
        self.res_dict = res_dict
        self.res_name = res_name


def all_subclasses(c):
    ret = set()
    to_add = c.__subclasses__()
    while len(to_add) > 0:
        i = to_add.pop()
        to_add.extend(i.__subclasses__())
        ret.add(i)

    return ret


Available_Problems = Resource(
    dict((e.name, e) for e in all_subclasses(AprxSampler) if hasattr(e, 'name')),
    "Problem"
)


Available_Policies = Resource(
    dict((e.name, e) for e in all_subclasses(TeachingPolicy) if hasattr(e, 'name')),
    "Algorithm"
)


def find_resources(names_list, resources_dict):
    result = []
    for name in names_list:
        if name not in resources_dict:
            raise Exception(resources_dict.res_name + " '" + name + "' not found")
        print(resources_dict.res_name + " '" + name + "' added to list")
        result.append(resources_dict.res_dict[name])
    return result
