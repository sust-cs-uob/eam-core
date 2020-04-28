import eam_core.util as util
import unittest
import os
from collections import namedtuple
from deepdiff import DeepDiff
from eam_core.yaml_runner import load_configuration, setup_parser
from eam_core.YamlLoader import YamlLoader
from types import SimpleNamespace
import yaml


def compare_objects(first_object, second_object, ignore_order=False, report_repetition=False, significant_digits=None,
                    verbose_level=2):
    """
    Return difference between two objects.
    :param first_object: first object to compare
    :param second_object: second object to compare
    :param ignore_order: ignore difference in order
    :param report_repetition: report when is repetition
    :param significant_digits: use to properly compare numbers(float arithmetic error)
    :param verbose_level: higher verbose level shows you more details - default 0.
    :return: difference between two objects
    """

    diff = DeepDiff(first_object, second_object, ignore_order=ignore_order,
                    report_repetition=report_repetition, significant_digits=significant_digits,
                    verbose_level=verbose_level)
    return diff


class MyTestCase(unittest.TestCase):

    def test_existing_ids(self):
        dict = {"IDs": True, "analysis_config": "dev", "comment": None, "docker": True, "documentation": False,
                "filetype": "pdf", "local": True, "sensitivity": False, "verbose": False,
                "yamlfile": "tests/models/existing_ids.yml"}
        args = SimpleNamespace(**dict)
        model_run_base_directory, simulation_run_description, yaml_struct = load_configuration(args)
        scenario = "default"
        model_output_directory = model_run_base_directory + f"/{scenario}"
        if not os.path.exists(model_output_directory):
            os.makedirs(model_output_directory)
        create_model_func, sim_control, yaml_struct = util.prepare_simulation(model_output_directory,
                                                                              simulation_run_description, yaml_struct,
                                                                              scenario, filename=args.yamlfile,
                                                                              IDs=args.IDs)
        loader = YamlLoader(version=yaml_struct.get('variant', 1))
        loader.parse_yaml_structure(yaml_struct, sim_control)
        old=open("tests/models/ci_v2copy.yml")
        new=open("tests/models/ci_v2.yml")
        returned_diff = compare_objects(yaml.load(old, Loader=yaml.FullLoader), yaml.load(new ,  Loader=yaml.FullLoader))
        old.close()
        new.close()
        assert(returned_diff == {})
        assert loader.id_map["process"] == {"CDN": 0, "Internet Network": 1, "Laptop": 2}
