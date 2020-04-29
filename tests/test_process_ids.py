import eam_core.util as util
import unittest
import os
from deepdiff import DeepDiff
from eam_core.yaml_runner import load_configuration
from eam_core.YamlLoader import YamlLoader
from types import SimpleNamespace
import yaml
from ruamel import yaml


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


def get_ids(file, flag=True):
    dict = {"IDs": flag, "analysis_config": "dev", "comment": None, "docker": True, "documentation": False,
            "filetype": "pdf", "local": True, "sensitivity": False, "verbose": False,
            "yamlfile": "tests/models/" + file + ".yml"}
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
    old = open("tests/models/" + file + "_copy.yml")
    old_struct = yaml.load(old, Loader=yaml.RoundTripLoader)
    new = open("tests/models/" + file + ".yml")
    new_struct = yaml.load(new, Loader=yaml.RoundTripLoader)
    diff = compare_objects(old_struct, new_struct)
    old.close()
    new.close()
    with open("tests/models/" + file + ".yml", 'w') as file:
        yaml.dump(old_struct, file, default_flow_style=False, Dumper=yaml.RoundTripDumper,
                  width=4096)
    return diff, loader.id_map


class TestProcessIDs(unittest.TestCase):

    def test_existing_ids(self):
        diff,id_map=get_ids("existing_ids")
        assert (diff == {})
        assert id_map["process"] == {"CDN": 0, "Internet Network": 1, "Laptop": 2}

    def test_some_existing_ids(self):
        diff, id_map = get_ids("some_existing_ids")
        assert (diff == {'dictionary_item_added': {"root['Processes'][1]['id']": 4, "root['Processes'][2]['id']": 5}})
        assert id_map["process"] == {"CDN": 3, "Internet Network": 4, "Laptop": 5}

    def test_no_existing_ids(self):
        diff, id_map = get_ids("no_existing_ids")
        print(diff)
        assert diff == {'dictionary_item_added': {"root['Processes'][0]['id']": 0, "root['Processes'][1]['id']": 1, "root['Processes'][2]['id']": 2}}
        assert id_map["process"] == {"CDN": 0, "Internet Network": 1, "Laptop": 2}

    def test_duplicate_ids(self):
        with self.assertRaises(Exception) as context:
            get_ids("duplicate_ids")
        self.assertTrue('Duplicate ID for process ' in str(context.exception))

    def test_no_id_flag(self):
        diff, id_map = get_ids("no_existing_ids", flag=False)
        assert diff == {}
        assert id_map["process"] == {}
